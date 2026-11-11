import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
from model.operators import (
    OPS, MSTF_NAS_SPACE_DARTS, MSTF_NAS_SPACE_PDC, MSTF_NAS_SPACE_REDUCT,
    DIDS, ReLUConvBN,
)
from DataSets import CRC5000, DataSet
from architect import Architect
import utils


parser = argparse.ArgumentParser("MedNAS-DARTS-Search")
parser.add_argument('--dataset', type=str, default='DermaMNIST', help='dataset name')
parser.add_argument('--data_path', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers (blocks)')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--normal_space', choices=['normal', 'pdc'], default='normal',
                    help='Normal Cell candidate pool: original Normal or proposed PDC space')
parser.add_argument('--downsample', choices=['reduction', 'dids'], default='reduction',
                    help='scale-transform module: searched original Reduction Cell or fixed DIDS')
args = parser.parse_args()

args.save = 'search-{}-{}-{}-{}'.format(
    args.dataset, args.normal_space, args.downsample, time.strftime("%Y%m%d-%H%M%S")
)
if not os.path.exists(args.save):
    os.makedirs(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

class MixedOp(nn.Module):
    def __init__(self, C, stride, ops_space_keys):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in ops_space_keys:
            op = OPS[primitive](C, False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))



class NormalCellSearch(nn.Module):
    def __init__(self, in_c, num_node=5, normal_space=MSTF_NAS_SPACE_PDC):
        super(NormalCellSearch, self).__init__()
        self.num_node = num_node
        self.mixed_ops = nn.ModuleList()

        for i in range(num_node):
            for j in range(i + 1):
                self.mixed_ops.append(MixedOp(in_c, 1, normal_space))

        self.out_c = in_c

    def forward(self, x, weights):
        states = [x]
        offset = 0

        node_outputs = []
        for i in range(self.num_node):
            current_edges_ops = []
            for j in range(i + 1):
                op = self.mixed_ops[offset + j]
                w = weights[offset + j]
                res = op(states[j], w)
                current_edges_ops.append(res)

            offset += (i + 1)
            s_i = sum(current_edges_ops)
            states.append(s_i)
            node_outputs.append(s_i)

        return sum(node_outputs) / self.num_node


class MixedOpReduction(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self._ops = nn.ModuleList([OPS[primitive](in_c, out_c, False) for primitive in MSTF_NAS_SPACE_REDUCT])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class ReductCellSearch(nn.Module):
    """Original searched Reduction Cell used in E1 and E2."""
    def __init__(self, in_c, num_node=4):
        super().__init__()
        if in_c % num_node:
            raise ValueError("in_c must be divisible by the Reduction Cell branch count.")
        self.num_node = num_node
        self.out_c = 2 * in_c
        branch_c = in_c // num_node
        self.branches = nn.ModuleList([MixedOpReduction(in_c, branch_c) for _ in range(num_node)])
        self.post_process = ReLUConvBN(in_c, self.out_c, kernel_s=1, stride=1, padding=0, affine=False)

    def forward(self, x, weights):
        return self.post_process(torch.cat([branch(x, weights[i]) for i, branch in enumerate(self.branches)], dim=1))


# Supernet Block
class BlockSearch(nn.Module):
    def __init__(self, in_c, n_n, n_r, normal_space, downsample):
        super(BlockSearch, self).__init__()
        self.nc = NormalCellSearch(in_c, n_n, normal_space)
        self.downsample_type = downsample
        if downsample == 'dids':
            self.downsample = DIDS(in_c, out_c=2 * in_c, reduction=4)
        else:
            self.downsample = ReductCellSearch(in_c, n_r)
        self.out_c = self.downsample.out_c

    def forward(self, x, alphas_normal, alphas_reduce=None):
        w_normal = F.softmax(alphas_normal, dim=-1)
        x = self.nc(x, w_normal)
        if self.downsample_type == 'dids':
            return self.downsample(x)
        return self.downsample(x, F.softmax(alphas_reduce, dim=-1))



class Network(nn.Module):
    def __init__(self, init_c, num_classes, layers, criterion, n_n=5, n_r=4,
                 normal_space='pdc', downsample='dids'):
        super(Network, self).__init__()
        self._C = init_c
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self.n_n = n_n
        self.n_r = n_r
        self.normal_space_name = normal_space
        self.downsample_type = downsample
        self.normal_space = {'normal': MSTF_NAS_SPACE_DARTS, 'pdc': MSTF_NAS_SPACE_PDC}[normal_space]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=init_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=init_c, affine=False),
        )

        self.blocks = nn.ModuleList()
        curr_c = init_c
        for i in range(layers):
            blk = BlockSearch(curr_c, n_n, n_r, self.normal_space, downsample)
            self.blocks.append(blk)
            curr_c = blk.out_c

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(curr_c, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion, self.n_n, self.n_r,
            self.normal_space_name, self.downsample_type,
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, self.alphas_normal, getattr(self, 'alphas_reduce', None))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        num_ops_normal = len(self.normal_space)
        k_normal = sum(range(1, self.n_n + 1))

        self.alphas_normal = Variable(1e-3 * torch.randn(k_normal, num_ops_normal).cuda(), requires_grad=True)

        self._arch_parameters = [self.alphas_normal]
        if self.downsample_type == 'reduction':
            self.alphas_reduce = Variable(
                1e-3 * torch.randn(self.n_r, len(MSTF_NAS_SPACE_REDUCT)).cuda(), requires_grad=True
            )
            self._arch_parameters.append(self.alphas_reduce)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # Parse Normal Cell as [ops_idx, in_idx].  ``none`` remains a normal
        # mixed-operation candidate during search, but cannot become the sole
        # retained operation of a discrete node.
        gene_normal = []
        offset = 0
        W_n = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
        none_idx = self.normal_space.index('none')

        for i in range(self.n_n):
            current_W = W_n[offset: offset + i + 1]
            non_none_W = current_W.copy()
            non_none_W[:, none_idx] = -np.inf
            idx_flat = np.argmax(non_none_W)
            best_edge_idx, best_op_idx = np.unravel_index(idx_flat, non_none_W.shape)

            gene_normal.append(int(best_op_idx))
            gene_normal.append(int(best_edge_idx))
            offset += (i + 1)

        if self.downsample_type == 'dids':
            return gene_normal
        W_r = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()
        return gene_normal + [int(np.argmax(W_r[i])) for i in range(self.n_r)]


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def prepare_target(target, dataset):
    """Move labels to CUDA while preserving ChestMNIST's [B, 14] targets."""
    target = Variable(target).cuda()
    if dataset == 'ChestMNIST':
        return target.float()
    # Single-label MedMNIST targets are conventionally [B, 1].  ``view(-1)``
    # also preserves a final evaluation batch with B=1 as a length-one vector.
    return target.long().view(-1)


def batch_accuracy(logits, target, dataset):
    """Return the existing training-style accuracy for one search batch."""
    if dataset == 'ChestMNIST':
        predictions = (torch.sigmoid(logits) > 0.5).float()
        # Multi-label ChestMNIST has no top-k definition.  This is the same
        # label-wise accuracy used by Train.py, reported as a percentage.
        return (predictions == target).float().mean() * 100.0
    return utils.accuracy(logits, target, topk=(1,))[0]


# ============================================================

# ============================================================
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch, warmup=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()


    valid_queue_iter = iter(valid_queue)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input).cuda()
        target = prepare_target(target, args.dataset)

        # Get valid batch for architecture step
        try:
            input_search, target_search = next(valid_queue_iter)
        except StopIteration:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = Variable(input_search).cuda()
        target_search = prepare_target(target_search, args.dataset)

   
        if not warmup:
            # ============================================================
   
            # ============================================================
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, epoch=epoch, n_epochs=args.epochs)


        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = batch_accuracy(logits, target, args.dataset)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = prepare_target(target, args.dataset)

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = batch_accuracy(logits, target, args.dataset)
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'ChestMNIST':
        # ChestMNIST is a 14-label multi-label task.  Its target must remain
        # [B, 14] and be optimized independently per label.
        criterion = nn.BCEWithLogitsLoss().cuda()
        logging.info('ChestMNIST multi-label mode: BCEWithLogitsLoss')
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    if args.dataset == 'CRC5000':
        # CRC-5000 is pre-partitioned per class as 60%/20%/20%.  The search
        # dataset is [3,000 train | 1,000 validation], so 0.75 preserves that
        # boundary and keeps the independent 1,000-image test set untouched.
        args.train_portion = 0.75
        Train_Data_All, Test_Data, num_class = CRC5000(
            size=150, purpose='search', seed=args.seed,
        )
        logging.info(
            'CRC-5000 protocol: 3000 train / 1000 valid / 1000 held-out test; '
            'search train_portion fixed to %.2f', args.train_portion,
        )
    else:
        Train_Data_All, Test_Data, num_class = DataSet[args.dataset](size=28)

    num_train = len(Train_Data_All)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        Train_Data_All, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        Train_Data_All, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)

    logging.info('ablation config: normal_space=%s, downsample=%s', args.normal_space, args.downsample)
    model = Network(
        args.init_channels, num_class, args.layers, criterion,
        normal_space=args.normal_space, downsample=args.downsample,
    ).cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    best_valid_acc = float('-inf')
    best_epoch = -1
    best_genotype = None

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # ============================================================

        # ============================================================
        # ----------------------------------------------------------------
 
        # ----------------------------------------------------------------
        is_warmup = epoch < 5
        if is_warmup:
            logging.info(f'Warmup Phase: Only Network Weights Updated (Epoch {epoch})')
        
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch=epoch, warmup=is_warmup)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # Export the discrete architecture selected by validation accuracy,
        # rather than unconditionally using the final epoch's architecture.
        # ``model.genotype()`` is called after this epoch's architecture update.
        current_genotype = model.genotype()
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_genotype = current_genotype

    if best_genotype is None:
        raise RuntimeError('Search completed without a validation result; no genotype can be exported.')

    logging.info(
        'final genotype = %s (best valid_acc %.4f at epoch %d)',
        best_genotype, best_valid_acc, best_epoch,
    )
    with open(os.path.join(args.save, 'genotype.txt'), 'w') as f:
        # Keep this file as a pure list so Train.py and ablation scripts can
        # pass it directly through --cell_genotype.
        f.write(str(best_genotype))
    with open(os.path.join(args.save, 'genotype_info.txt'), 'w') as f:
        f.write(f'best_valid_acc={best_valid_acc:.6f}\n')
        f.write(f'best_epoch_zero_based={best_epoch}\n')
        f.write(f'best_epoch_one_based={best_epoch + 1}\n')
        f.write(f'genotype={best_genotype}\n')

    utils.save(model, os.path.join(args.save, 'weights.pt'))


if __name__ == '__main__':
    main()
