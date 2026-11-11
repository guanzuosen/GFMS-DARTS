import argparse
import ast
import os
import json
from pathlib import Path
import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
from datetime import datetime

from model.net_cell import MSTF_NAS_NETWORK
from DataSets import *
import warnings

try:
    import psutil
except ImportError:
    psutil = None

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from scipy.special import softmax

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ==========================================

# ==========================================
# PDC operation indices: 0=PDC-Sep3, 1=avgpool3, 2=Sep3,
# 3=Sep5, 4=skip, 5=none/zero, 6=dil3, 7=dil5.
# Pass --cell_genotype to train the architecture exported by a search run.
DEFAULT_CELL_GENOTYPE =   [2,0, 3,1, 1,0, 6,2, 0,4]

# These two datasets use task-specific label definitions and are deliberately
# excluded from the single-label macro metrics requested for the other tasks.
METRICS_EXCLUDED_DATASETS = {"ChestMNIST", "TissueMNIST"}


def log_string(log_file, content):
    print(content)  
    with open(log_file, 'a') as f: 
        f.write(content + '\n')


def log_final_summary(log_file, content):
    """Write the final summary at most once for each training log file."""
    summary_marker = "训练完成！最终结果总结："
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            if summary_marker in f.read():
                return
    log_string(log_file, content)


def Train(train_loader, model, criterion, optimizer, device, new_iter, num_class, Args):
   
    model.train()

    Loss_list = []
   
    Acc_list = [[], []]
    now_iter = 0

    for batch_idx, (Images, Labels) in enumerate(train_loader):
        Images, Labels = Images.to(device), Labels.to(device)

    
        # print(f"[DEBUG] Batch {batch_idx}:")
        # print(f"[DEBUG]   Images shape: {Images.shape}")
        # print(f"[DEBUG]   Labels shape: {Labels.shape}")
        # print(f"[DEBUG]   Labels dtype: {Labels.dtype}")
        # print(f"[DEBUG]   Labels unique values: {torch.unique(Labels)}")
        # # ================================

      
        if Args.data == "ChestMNIST":
          
            Labels = Labels.float()
        else:
        
            Labels = Labels.view(-1).long()

        optimizer.zero_grad()
   
        Logits = model(Images)

      
        # print(f"[DEBUG]   Logits shape: {Logits.shape}")
        # print(f"[DEBUG]   Logits dtype: {Logits.dtype}")
        # # ================================

        
        if Logits.shape[0] != Labels.shape[0]:
            print(f"[ERROR] Batch size mismatch! Logits: {Logits.shape[0]}, Labels: {Labels.shape[0]}")
            print(f"[ERROR] This batch will be skipped.")
            continue  # 跳过这个batch

        Loss = criterion(Logits, Labels)
      
        Loss.backward()
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
       
        optimizer.step()

 
        Batch_Size = Images.size(0)

        if Args.data == "ChestMNIST":
    
            Pred_probs = torch.sigmoid(Logits)
         
            Pred_binary = (Pred_probs > 0.5).float()

          
            label_correct = (Pred_binary == Labels).float()
           
            label_accuracies = label_correct.mean(dim=0)  # 形状: [num_labels]
       
            Acc_1 = label_accuracies.mean().item() * 100
            Acc_5 = Acc_1  

        
            Acc_list[0].append(Acc_1)
            Acc_list[1].append(Acc_5)



        else:
   
            topk = (1, num_class // 2)
            
            _, Pred = torch.topk(input=Logits.data, k=max(topk), dim=1, largest=True, sorted=True)
          
            Pred = Pred.T  # Pred.size() = [max(topk), batch_size]
           
            Correct = torch.eq(Pred, Labels.data.view(1, -1))
           
            for index, k in enumerate(topk):
                Correct_k = Correct[0:k].contiguous().view(-1).float().sum()
                Acc_list[index].append(Correct_k.mul(100.0 / Batch_Size).item())
            
        
            # Acc_1 = Acc_list[0][-1] if Acc_list[0] else 0
            # Acc_5 = Acc_list[1][-1] if Acc_list[1] else 0

        Loss_list.append(Loss.item())

        now_iter = now_iter + 1
        if now_iter >= new_iter: break

   
    avg_loss = sum(Loss_list) / len(Loss_list) if Loss_list else 0.0

  
    if Args.data == "ChestMNIST" and Acc_list[0]:
        avg_acc1 = sum(Acc_list[0]) / len(Acc_list[0])
        avg_acc5 = avg_acc1  
    else:
        avg_acc1 = sum(Acc_list[0]) / len(Acc_list[0]) if Acc_list[0] else 0.0
        avg_acc5 = sum(Acc_list[1]) / len(Acc_list[1]) if Acc_list[1] else 0.0

    return [avg_loss, avg_acc1, avg_acc5]


def calculate_auc(logits, labels, num_class, is_multilabel=False):
 
    try:
        if is_multilabel:
          
            probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
            labels = labels.astype(int)
            auc_scores = []
            for i in range(num_class):
                try:
                    auc_i = roc_auc_score(labels[:, i], probabilities[:, i])
                    auc_scores.append(auc_i)
                except ValueError:
                    auc_scores.append(0.5)  
            auc = np.mean(auc_scores)
        else:
        
            probabilities = softmax(logits, axis=1)
            labels = labels.astype(int)

   
            if num_class == 2:
                auc = roc_auc_score(labels, probabilities[:, 1])
            else:
               
                auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        return auc
    except Exception as e:
        print(f"AUC error: {e}")
        return 0.0


def calculate_classification_metrics(logits, labels, num_class):
    """Compute four single-label test metrics with a uniform macro protocol.

    F1, precision and recall are sklearn macro averages across all classes.
    Specificity is also macro averaged, with each class treated as positive in
    a one-vs-rest confusion matrix.  Returned values are fractions in [0, 1],
    matching the reporting convention used in the comparison table.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if logits.ndim != 2 or logits.shape[0] != labels.size:
        raise ValueError(
            f"Metric inputs must be logits [N, C] and labels [N]; received "
            f"{logits.shape} and {labels.shape}."
        )
    if logits.shape[1] != num_class:
        raise ValueError(
            f"Expected {num_class} logit columns, but received {logits.shape[1]}."
        )

    predictions = logits.argmax(axis=1)
    class_ids = np.arange(num_class)
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = matrix.sum() - true_positive - false_positive - false_negative
    specificity_per_class = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.zeros(num_class, dtype=float),
        where=(true_negative + false_positive) != 0,
    )

    return {
        "f1_macro": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(
            labels, predictions, labels=class_ids, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "specificity_macro_ovr": float(specificity_per_class.mean()),
    }


def Test(test_loader, model, criterion, device, num_class, Args):
    model.eval()
    Loss_list = []
    Acc_list = [[], []]
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (Images, Labels) in enumerate(test_loader):
            Images, Labels = Images.to(device), Labels.to(device)

            # 对于多标签分类（如ChestMNIST），需要调整标签格式
            if Args.data == "ChestMNIST":
                Labels = Labels.float()
            else:
                Labels = Labels.view(-1).long()

            Logits = model(Images)
            Loss = criterion(Logits, Labels)

            all_logits.append(Logits.cpu().numpy())
            all_labels.append(Labels.cpu().numpy())


            Batch_Size = Images.size(0)

            if Args.data == "ChestMNIST":
              
           
                Pred_probs = torch.sigmoid(Logits)
               
                Pred_binary = (Pred_probs > 0.5).float()

          
                label_correct = (Pred_binary == Labels).float()
                label_accuracies = label_correct.mean(dim=0)  # 形状: [num_labels]
                Acc_1 = label_accuracies.mean().item() * 100
                Acc_5 = Acc_1  

                Acc_list[0].append(Acc_1)
                Acc_list[1].append(Acc_5)
            else:
            
                topk = (1, num_class // 2)
                _, Pred = torch.topk(input=Logits.data, k=max(topk), dim=1, largest=True, sorted=True)
                Pred = Pred.T  # Pred.size() = [max(topk), batch_size]
                Correct = torch.eq(Pred, Labels.data.view(1, -1))
                for index, k in enumerate(topk):
                    Correct_k = Correct[0:k].contiguous().view(-1).float().sum()
                    Acc_list[index].append(Correct_k.mul(100.0 / Batch_Size).item())

            Loss_list.append(Loss.item())


    if len(all_logits) == 0:
        print("警告: 测试集没有数据或模型没有输出!")
        return [0.0, 0.0, 0.0, 0.0, None]

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels) if all_labels[0].ndim > 1 else np.hstack(all_labels)

    
    is_multilabel = (Args.data == "ChestMNIST")
    auc_score = calculate_auc(all_logits, all_labels, num_class, is_multilabel)
    classification_metrics = None
    if Args.data not in METRICS_EXCLUDED_DATASETS:
        classification_metrics = calculate_classification_metrics(all_logits, all_labels, num_class)


    avg_loss = sum(Loss_list) / len(Loss_list) if Loss_list else 0.0

  
    if Args.data == "ChestMNIST" and Acc_list[0]:
        avg_acc1 = sum(Acc_list[0]) / len(Acc_list[0])
        avg_acc5 = avg_acc1  
    else:
        avg_acc1 = sum(Acc_list[0]) / len(Acc_list[0]) if Acc_list[0] else 0.0
        avg_acc5 = sum(Acc_list[1]) / len(Acc_list[1]) if Acc_list[1] else 0.0

    return [avg_loss, avg_acc1, avg_acc5, auc_score, classification_metrics]

def plot_training_curves(train_losses, test_losses, train_acc1, test_acc1, train_acck, test_acck, test_aucs, save_path,
                         k_value=5):

    if plt is None:
        raise ImportError("绘制训练曲线需要 matplotlib；请先安装该依赖。")
    epochs = range(1, len(train_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))


    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

 
    ax2.plot(epochs, train_acc1, 'b-', label='Train Top-1 Acc', linewidth=2)
    ax2.plot(epochs, test_acc1, 'r-', label='Test Top-1 Acc', linewidth=2)
    ax2.set_title('Top-1 Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    ax3.plot(epochs, train_acck, 'b-', label=f'Train Top-{k_value} Acc', linewidth=2)
    ax3.plot(epochs, test_acck, 'r-', label=f'Test Top-{k_value} Acc', linewidth=2)
    ax3.set_title(f'Top-{k_value} Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


    ax4.plot(epochs, test_aucs, 'g-', label='Test AUC', linewidth=2)
    ax4.set_title('Test AUC')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存至: {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='TissueMNIST', type=str, help="Dataset name")
    parser.add_argument('--data_num', default=102400000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--now_times', default=1, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--seed', default=2, type=int, help="random seed")
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--arch_file', default='./logs_search1_GA', help="File containing the best architecture")
 
    parser.add_argument('--type', default='train', help="Experiment type") 
    parser.add_argument('--normal_space', choices=['normal', 'pdc'], default='pdc',
                        help='must match the Normal Cell search-space used to export the genotype')
    parser.add_argument('--downsample', choices=['reduction', 'dids'], default='dids',
                        help='must match the scale-transform module used during search')
    parser.add_argument('--cell_genotype', type=str, default=None,
                        help='one Block genotype, e.g. "[0,0,1,0,...]"; repeated for all blocks')
    parser.add_argument('--transfer_genotype_file', type=str, default=None,
                        help='PathMNIST genotype.txt used for LC25000 transfer; defaults to the latest completed PathMNIST search')
    parser.add_argument('--n_nodes', type=int, default=5, help='Normal Cell node count')
    parser.add_argument('--n_reduce_nodes', type=int, default=4, help='Reduction Cell branch count')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of Normal Cell + downsample blocks')
    parser.add_argument('--init_channels', type=int, default=128, help='stem output channels')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='save/overwrite the best Test Top-1 checkpoint during training')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='directory for a best-model checkpoint when --save_checkpoint is set')
    parser.add_argument('--source_dataset', default=None, type=str,
                        help='optional architecture source dataset recorded in the checkpoint metadata')
    
    args = parser.parse_args()

    # ==========================================
  
    # ==========================================
    dataset_configs = {
        'PathMNIST': {'batch_size': 512, 'lr': 0.05},
        'ChestMNIST': {'batch_size': 512, 'lr': 0.05},
        'DermaMNIST': {'batch_size': 128, 'lr': 0.025},
        'OCTMNIST': {'batch_size': 512, 'lr': 0.05},
        'PneumoniaMNIST': {'batch_size': 128, 'lr': 0.0005},
        'RetinaMNIST': {'batch_size': 128, 'lr': 0.001},
        'BreastMNIST': {'batch_size': 128, 'lr': 0.001},
        'BloodMNIST': {'batch_size': 128, 'lr': 0.025},
        'TissueMNIST': {'batch_size': 128, 'lr': 0.025},
        'OrganAMNIST': {'batch_size': 256, 'lr': 0.025},
        'OrganCMNIST': {'batch_size': 128, 'lr': 0.025},
        'OrganSMNIST': {'batch_size': 128, 'lr': 0.025},
        # Keep the CRC-5000 optimization configuration identical to
        # PneumoniaMNIST, as required by the experiment protocol.
        'CRC5000': {'batch_size': 64, 'lr': 0.0005},
        'LC25000': {'batch_size': 16, 'lr': 0.0005},
    }

    
    if args.data in dataset_configs:
        config = dataset_configs[args.data]
        args.batch_size = config['batch_size']
        args.lr = config['lr']
        print(f"自动配置 {args.data}: batch_size={args.batch_size}, lr={args.lr}")
    else:
        print(f"警告: 数据集 {args.data} 不在预设配置中，使用默认参数")

    return args


def resolve_lc25000_transfer_genotype(args):
    """Resolve a PathMNIST genotype without ever loading PathMNIST weights."""
    if args.data != 'LC25000':
        return args.cell_genotype, None
    if args.normal_space != 'pdc':
        raise ValueError('LC25000 PathMNIST transfer requires --normal_space pdc.')
    if args.downsample != 'dids':
        raise ValueError('LC25000 transfer mode requires --downsample dids.')
    if args.cell_genotype is not None and args.transfer_genotype_file is not None:
        raise ValueError('Use either --cell_genotype or --transfer_genotype_file for LC25000, not both.')

    if args.cell_genotype is not None:
        return args.cell_genotype, 'provided via --cell_genotype (PathMNIST source)'

    if args.transfer_genotype_file is not None:
        source_path = Path(args.transfer_genotype_file)
    else:
        repo_root = Path(__file__).resolve().parent
        candidates = sorted(
            repo_root.glob('search-PathMNIST-*/genotype.txt'),
            key=lambda path: path.parent.name,
        )
        if not candidates:
            raise FileNotFoundError(
                'No PathMNIST search genotype.txt was found. Pass --transfer_genotype_file explicitly.'
            )
        source_path = candidates[-1]
    if not source_path.is_file():
        raise FileNotFoundError(f'PathMNIST transfer genotype file does not exist: {source_path}')
    raw_genotype = source_path.read_text(encoding='utf-8').strip()
    if not raw_genotype:
        raise RuntimeError(f'PathMNIST transfer genotype file is empty: {source_path}')
    return raw_genotype, str(source_path.resolve())


def parse_cell_genotype(raw_genotype, args):
    """Parse and validate one searched Block genotype before repeating it."""
    if raw_genotype is None:
        genotype = DEFAULT_CELL_GENOTYPE
    else:
        try:
            genotype = ast.literal_eval(raw_genotype)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("--cell_genotype must be a Python-style integer list.") from exc
    if not isinstance(genotype, list) or not all(isinstance(value, int) for value in genotype):
        raise ValueError("--cell_genotype must be a list of integers.")

    expected_len = 2 * args.n_nodes + (args.n_reduce_nodes if args.downsample == 'reduction' else 0)
    if len(genotype) != expected_len:
        raise ValueError(
            f"Expected {expected_len} values for one Block with downsample={args.downsample}, "
            f"but received {len(genotype)}."
        )
    return genotype


def limit_cpu():

    if psutil is None:
        print("未安装 psutil，跳过 CPU 亲和性设置。")
        return
    try:
  
        p = psutil.Process()
        p.cpu_affinity([0,1,2 ,3]) 
    except Exception as e:
        print(f"设置CPU亲和性失败 (可能是系统不支持): {e}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    limit_cpu()
    

    Args = parse_arguments()
    Raw_Genotype, Transfer_Genotype_Source = resolve_lc25000_transfer_genotype(Args)
    Cell_Genotype = parse_cell_genotype(Raw_Genotype, Args)
    Genotype = Cell_Genotype * Args.num_blocks
    Is_LC25000_Transfer = Args.data == 'LC25000'
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    torch.cuda.manual_seed(Args.seed)
    torch.cuda.manual_seed_all(Args.seed)

    best_test_acc1 = 0.0
    best_test_acck = 0.0
    best_test_auc = 0.0
    best_epoch_acc1 = 0
    best_epoch_acck = 0
    best_epoch_auc = 0


    train_losses = []
    test_losses = []
    train_acc1_list = []
    test_acc1_list = []
    train_acck_list = []
    test_acck_list = []
    test_auc_list = []

 
    if torch.cuda.is_available():
        if Args.gpu < 0 or Args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu must be in [0, {torch.cuda.device_count() - 1}], received {Args.gpu}.")
        torch.cuda.set_device(Args.gpu)
        torch.cuda.empty_cache()
        device = torch.device(f'cuda:{Args.gpu}')
    else:
        device = torch.device('cpu')
    torch.backends.cudnn.benchmark = torch.backends.cudnn.enabled
    
   
    if not os.path.exists('./log'):
        os.makedirs('./log', exist_ok=True)
    if not os.path.exists(Args.arch_file):
        os.makedirs(Args.arch_file, exist_ok=True)
        
  
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    Text_Log_File = f"./log/train-{Args.data}-{timestamp}.txt"
    Checkpoint_Path = None
    if Args.save_checkpoint:
        os.makedirs(Args.checkpoint_dir, exist_ok=True)
        Checkpoint_Path = os.path.join(
            Args.checkpoint_dir,
            f"checkpoint-{Args.data}-{Args.normal_space}-{Args.downsample}-"
            f"seed{Args.seed}-{timestamp}-best_test_top1.pt",
        )
    
   
    log_string(Text_Log_File, "="*60)
    log_string(Text_Log_File, f"Experiment Start: {timestamp}")
    log_string(Text_Log_File, f"Dataset: {Args.data}")
    log_string(Text_Log_File, f"GPU: {Args.gpu if torch.cuda.is_available() else 'cpu'}")
    log_string(Text_Log_File, f"Training Seed: {Args.seed}")
    log_string(Text_Log_File, f"Ablation config: normal_space={Args.normal_space}, downsample={Args.downsample}")
    log_string(Text_Log_File, f"Cell genotype: {Cell_Genotype}")
    if Checkpoint_Path is not None:
        log_string(Text_Log_File, "Checkpoint selection: best Test Top-1 accuracy")
        log_string(Text_Log_File, f"Checkpoint path: {Checkpoint_Path}")
    if Is_LC25000_Transfer:
        log_string(Text_Log_File, "Mode: Transfer")
        log_string(Text_Log_File, "Source Dataset: PathMNIST")
        log_string(Text_Log_File, "Target Dataset: LC25000")
        log_string(Text_Log_File, "Transfer Genotype: True")
        log_string(Text_Log_File, "Transfer Weights: False")
        log_string(Text_Log_File, f"Transfer Genotype File: {Transfer_Genotype_Source}")
        log_string(Text_Log_File, "Evaluation Protocol: Regular train-test evaluation every epoch")
        log_string(Text_Log_File, "Validation Split: Fixed but unused in regular training mode")
    log_string(Text_Log_File, f"gpu?\t{device}")
    log_string(Text_Log_File, f"Is cuda  available?\t{torch.cuda.is_available()}")
    log_string(Text_Log_File, f"Is cudnn available?\t{torch.backends.cudnn.enabled}")
    log_string(Text_Log_File, "="*60 + "\n")

    # CRC-5000 uses its fixed, class-stratified 60/20/20 partition.  Final
    # training receives only the 3,000-image training split and evaluates on
    # the independent 1,000-image test split.
    if Args.data == 'CRC5000':
        Train_Data, Test_Data, num_class = CRC5000(
            size=150, purpose='train', seed=Args.seed,
        )
        print('CRC-5000 protocol: 3000 train / 1000 valid (search only) / 1000 test')
        Valid_Data = None
        Split_File = None
    elif Is_LC25000_Transfer:
        Train_Data, Valid_Data, Test_Data, num_class, Split_File = LC25000(size=224)
        print('LC25000 regular protocol: fixed 20000 train / 2500 valid (unused) / 2500 test')
        print(f'LC25000 fixed split file: {Split_File}')
        log_string(Text_Log_File, 'LC25000 split: 20000 train / 2500 validation (unused) / 2500 test')
        log_string(Text_Log_File, f'LC25000 split file: {Split_File}')
    else:
        Train_Data, Test_Data, num_class = DataSet[Args.data](size=28)
        Valid_Data = None
        Split_File = None
    Train_Loader = torch.utils.data.DataLoader(dataset=Train_Data, batch_size=Args.batch_size, shuffle=True,
                                               drop_last=True, pin_memory=True, num_workers=2)
    if Is_LC25000_Transfer:
        Test_Loader = torch.utils.data.DataLoader(
            dataset=Test_Data, batch_size=Args.batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=2,
        )
        Valid_Loader = None
    else:
        Valid_Loader = None
        Test_Loader = torch.utils.data.DataLoader(dataset=Test_Data, batch_size=Args.batch_size, shuffle=False,
                                                  # Keep all 1,000 held-out CRC-5000 samples; its final batch
                                                  # has 104 images when batch_size=128.
                                                  drop_last=(Args.data != 'CRC5000'),
                                                  pin_memory=True, num_workers=2)
    Args.new_iter = min(Args.data_num // Args.batch_size, len(Train_Loader))

 
    # # print(f"[DEBUG] Train dataset size: {len(Train_Data)}")
    # # print(f"[DEBUG] Test dataset size: {len(Test_Data)}")
    # # print(f"[DEBUG] Train loader batch size: {Train_Loader.batch_size}")
    # # print(f"[DEBUG] Test loader batch size: {Test_Loader.batch_size}")
    # # print(f"[DEBUG] Number of classes: {num_class}")


    # # for images, labels in Train_Loader:
    # #     print(f"[DEBUG] First train batch - images: {images.shape}, labels: {labels.shape}")
    # #     print(f"[DEBUG] First train batch - labels dtype: {labels.dtype}")
    # #     break
    # #
  
    # # for images, labels in Test_Loader:
    # #     print(f"[DEBUG] First test batch - images: {images.shape}, labels: {labels.shape}")
    # #     print(f"[DEBUG] First test batch - labels dtype: {labels.dtype}")
    # #     break

    # Log Config
    # Log_File = f"./log/{Args.data}_{Args.type}_{Args.now_times}.json"

    # os.makedirs(os.path.dirname(Log_File), exist_ok=True)

    # Log Config
    Log_File = (
        f"{Args.arch_file}/{Args.data}_{Args.normal_space}_{Args.downsample}_"
        f"{Args.type}_{Args.now_times}.json"
    )


    if os.path.exists(Log_File):
        msg = f"警告: 日志文件 {Log_File} 已存在，将被覆盖"
        log_string(Text_Log_File, msg)
    else:
        msg = f"将创建新的日志文件: {Log_File}"
        log_string(Text_Log_File, msg)

    Log_Data = {
        'dataset': Args.data,
        'genotype': Genotype,
        'ablation_config': {
            'normal_space': Args.normal_space,
            'downsample': Args.downsample,
            'n_nodes': Args.n_nodes,
            'n_reduce_nodes': Args.n_reduce_nodes,
            'num_blocks': Args.num_blocks,
            'init_channels': Args.init_channels,
        },
        'num_class': num_class,
        'training_parameters': {
            'batch_size': Args.batch_size,
            'max_epoch': Args.max_epoch,
            'learning_rate': Args.lr,
            'optimizer': 'SGD',
            'scheduler': 'CosineAnnealingLR'
        },
        'Train': ['float', []],
        'Test': ['float', []],
    }
    if Args.data not in METRICS_EXCLUDED_DATASETS:
        Log_Data['Test_Classification_Metrics'] = ['macro_fraction', []]
    if Is_LC25000_Transfer:
        Log_Data['transfer'] = {
            'mode': 'Transfer',
            'source_dataset': 'PathMNIST',
            'target_dataset': 'LC25000',
            'transfer_genotype': True,
            'transfer_weights': False,
            'source_genotype_file': Transfer_Genotype_Source,
            'fixed_split_file': str(Split_File),
            'evaluation_protocol': 'regular_train_test_each_epoch',
            'validation_split_used': False,
        }

    # Training Config
    Network = MSTF_NAS_NETWORK(
        num_class=num_class, genotype=Genotype, n_n=Args.n_nodes, num_block=Args.num_blocks,
        init_c=Args.init_channels, normal_space=Args.normal_space,
        downsample=Args.downsample, n_r=Args.n_reduce_nodes,
    ).to(device)
    # Keep DataParallel's primary device aligned with --gpu; otherwise it defaults to cuda:0.
    if device.type == 'cuda':
        Network = nn.DataParallel(Network, device_ids=[Args.gpu], output_device=Args.gpu).to(device)
    Optimizer = torch.optim.SGD(params=Network.parameters(), lr=Args.lr, momentum=0.9, weight_decay=0)
    # Optimizer = torch.optim.Adam(params=Network.parameters(), lr=Args.lr, weight_decay=3e-4)

    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=Optimizer, T_max=Args.max_epoch)
    #Scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=Optimizer, gamma=0.9)  # PneumoniaMNIST


    if Args.data == "ChestMNIST":
        
        Criterion = nn.BCEWithLogitsLoss(reduction='mean')
        log_string(Text_Log_File, f"使用BCEWithLogitsLoss进行多标签分类")
    else:
   
        Criterion = nn.CrossEntropyLoss(reduction='mean')
        log_string(Text_Log_File, f"使用CrossEntropyLoss进行单标签分类")

    log_string(Text_Log_File, "\nStart Training...\n")
    log_string(Text_Log_File, "{:<5} | {:<20} | {:<15} | {:<15} | {:<15} | {:<10}".format(
            "Epoch", "Time", "Loss(Tr/Te)", "Acc1(Tr/Te)", "AccK(Tr/Te)", "AUC"))
    log_string(Text_Log_File, "-"*90)
    best_test_metrics = None
    best_metrics_epoch = 0

    for epoch in range(Args.max_epoch):
        Train_Loss, Train_Acc_1, Train_Acc_5 = Train(
            train_loader=Train_Loader, model=Network, criterion=Criterion,
            optimizer=Optimizer, device=device, new_iter=Args.new_iter,
            num_class=num_class, Args=Args,
        )
        Test_Loss, Test_Acc_1, Test_Acc_5, Test_AUC, Test_Metrics = Test(
            test_loader=Test_Loader, model=Network, criterion=Criterion,
            device=device, num_class=num_class, Args=Args,
        )
        Scheduler.step()

       
        train_losses.append(Train_Loss)
        test_losses.append(Test_Loss)
        train_acc1_list.append(Train_Acc_1)
        test_acc1_list.append(Test_Acc_1)
        train_acck_list.append(Train_Acc_5)
        test_acck_list.append(Test_Acc_5)
        test_auc_list.append(Test_AUC)

        if Test_Acc_1 > best_test_acc1:
            best_test_acc1 = Test_Acc_1
            best_epoch_acc1 = epoch + 1
            best_test_metrics = Test_Metrics
            best_metrics_epoch = epoch + 1
            if Checkpoint_Path is not None:
                model_to_save = Network.module if isinstance(Network, nn.DataParallel) else Network
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'dataset': Args.data,
                    'source_dataset': Args.source_dataset,
                    'genotype': Genotype,
                    'cell_genotype': Cell_Genotype,
                    'ablation_config': {
                        'normal_space': Args.normal_space,
                        'downsample': Args.downsample,
                        'n_nodes': Args.n_nodes,
                        'n_reduce_nodes': Args.n_reduce_nodes,
                        'num_blocks': Args.num_blocks,
                        'init_channels': Args.init_channels,
                    },
                    'num_class': num_class,
                    'training_parameters': {
                        'batch_size': Args.batch_size,
                        'max_epoch': Args.max_epoch,
                        'learning_rate': Args.lr,
                        'optimizer': 'SGD',
                        'scheduler': 'CosineAnnealingLR',
                    },
                    'seed': Args.seed,
                    'epoch': epoch + 1,
                    'selection_metric': 'test_top1_accuracy',
                    'selection_value': float(Test_Acc_1),
                    'test_auc_at_selection': float(Test_AUC),
                }, Checkpoint_Path)
        if Test_Acc_5 > best_test_acck:
            best_test_acck = Test_Acc_5
            best_epoch_acck = epoch + 1
        if Test_AUC > best_test_auc:
            best_test_auc = Test_AUC
            best_epoch_auc = epoch + 1

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_content = "{:<5} | {:<20} | {:.4f}/{:.4f}   | {:.2f}/{:.2f}     | {:.2f}/{:.2f}     | {:.4f}".format(
            epoch, current_time, Train_Loss, Test_Loss, Train_Acc_1, Test_Acc_1,
            Train_Acc_5, Test_Acc_5, Test_AUC,
        )
        log_string(Text_Log_File, log_content)
        if Test_Metrics is not None:
            log_string(
                Text_Log_File,
                "  Test macro metrics | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | "
                "Specificity: {:.4f}".format(
                    Test_Metrics['f1_macro'], Test_Metrics['precision_macro'],
                    Test_Metrics['recall_macro'], Test_Metrics['specificity_macro_ovr'],
                ),
            )

        Log_Data['Train'][1].append([Train_Loss, Train_Acc_1, Train_Acc_5])
        Log_Data['Test'][1].append([Test_Loss, Test_Acc_1, Test_Acc_5, Test_AUC])
        if Test_Metrics is not None:
            Log_Data['Test_Classification_Metrics'][1].append(Test_Metrics)
        with open(Log_File, 'w') as Log:
            json.dump(Log_Data, Log, indent=4)

    if best_test_metrics is not None:
        Log_Data['best_test_classification_metrics'] = {
            'epoch': best_metrics_epoch,
            'selection_metric': 'test_top1_accuracy',
            **best_test_metrics,
        }
        with open(Log_File, 'w') as Log:
            json.dump(Log_Data, Log, indent=4)

    summary = (
        "\n" + "=" * 80 + "\n"
        "训练完成！最终结果总结：\n"
        f"最佳Test Top-1准确率: {best_test_acc1:.4f}% (第 {best_epoch_acc1} 轮)\n"
        f"最佳Test Top-{num_class // 2}准确率: {best_test_acck:.2f}% (第 {best_epoch_acck} 轮)\n"
        f"最佳Test AUC: {best_test_auc:.4f} (第 {best_epoch_auc} 轮)\n"
        "=" * 80 + "\n"
    )
    if best_test_metrics is not None:
        summary = (
            summary.rstrip("\n") + "\n"
            f"峰值Test Top-1对应四项宏平均指标 (第 {best_metrics_epoch} 轮):\n"
            f"Test F1: {best_test_metrics['f1_macro']:.4f}\n"
            f"Test Precision: {best_test_metrics['precision_macro']:.4f}\n"
            f"Test Recall: {best_test_metrics['recall_macro']:.4f}\n"
            f"Test Specificity: {best_test_metrics['specificity_macro_ovr']:.4f}\n"
            + "=" * 80 + "\n"
        )
    log_final_summary(Text_Log_File, summary)
    
    print(f"TXT日志已保存至: {Text_Log_File}")
    print(f"JSON日志已保存至: {Log_File}")
    if Checkpoint_Path is not None:
        print(f"Best Test Top-1 checkpoint已保存至: {Checkpoint_Path}")

   