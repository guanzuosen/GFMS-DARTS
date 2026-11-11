import medmnist
import torchvision
import random
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

__all__   = ["DataSet", "CRC5000", "LC25000"]
Data_Path = '/home/gzs/AZNAS/MedNAS/medmnist/'
CRC5000_PATH = Path(__file__).resolve().parent / 'Kather_texture_2016_image_tiles_5000'
CRC5000_CLASSES = (
    '01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO',
    '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY',
)
CRC5000_NUM_CLASSES = len(CRC5000_CLASSES)
LC25000_PATH = Path(__file__).resolve().parent / 'LC25000'
LC25000_CLASSES = ('colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc')
LC25000_CLASS_PATHS = {
    'colon_aca': Path('colon_image_sets') / 'colon_aca',
    'colon_n': Path('colon_image_sets') / 'colon_n',
    'lung_aca': Path('lung_image_sets') / 'lung_aca',
    'lung_n': Path('lung_image_sets') / 'lung_n',
    'lung_scc': Path('lung_image_sets') / 'lung_scc',
}
LC25000_SPLIT_FILE = 'lc25000_stratified_split_80_10_10.json'
LC25000_SPLIT_SEED = 20260826
LC25000_NUM_CLASSES = len(LC25000_CLASSES)


DataSet = {
    # MedMNIST
    "PathMNIST"     : lambda size=28 : MedMNIST2D(size=size, name="PathMNIST"),
    "ChestMNIST"     : lambda size=28 : MedMNIST2D(size=size, name="ChestMNIST"),
    "DermaMNIST"    : lambda size=28 : MedMNIST2D(size=size, name="DermaMNIST"),
    "OCTMNIST"      : lambda size=28 : MedMNIST2D(size=size, name="OCTMNIST"),
    "PneumoniaMNIST": lambda size=28 : MedMNIST2D(size=size, name="PneumoniaMNIST"),
    "RetinaMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="RetinaMNIST"),
    "BreastMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="BreastMNIST"),
    "BloodMNIST"    : lambda size=28 : MedMNIST2D(size=size, name="BloodMNIST"),
    "TissueMNIST"    : lambda size=28 : MedMNIST2D(size=size, name="TissueMNIST"),
    "OrganAMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganAMNIST"),
    "OrganCMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganCMNIST"),
    "OrganSMNIST"   : lambda size=28 : MedMNIST2D(size=size, name="OrganSMNIST"),
    # CRC-5000 is a folder-based pathology dataset, not a MedMNIST archive.
    # The default purpose is the final 3,000-image training split plus the
    # independent 1,000-image test split.  Search.py requests purpose='search'
    # to obtain the 3,000/1,000 train/validation concatenation instead.
    "CRC5000"       : lambda size=150 : CRC5000(size=size, purpose='train', seed=0),
}
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
    }

def MedMNIST2D(size, name:str):
    MEAN, STD = [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]  # Use Cifar-10 statistics
    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        torchvision.transforms.AugMix(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    valid_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(), 
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    if name == "PathMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.PathMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.PathMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "ChestMNIST":
        # train_transform = transforms.Compose([
        #     transforms.Lambda(lambda image: image.convert('RGB')),
        #     torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=MEAN, std=STD)
        # ])
        Train_Data = medmnist.ChestMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.ChestMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)

    elif name == "DermaMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.0),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.DermaMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.DermaMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "OCTMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OCTMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.OCTMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "PneumoniaMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.PneumoniaMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.PneumoniaMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "RetinaMNIST":
        Train_Data = medmnist.RetinaMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.RetinaMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "BreastMNIST":
        Train_Data = medmnist.BreastMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.BreastMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "BloodMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.BloodMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.BloodMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "TissueMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.TissueMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.TissueMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "OrganAMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganAMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.OrganAMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "OrganCMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganCMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.OrganCMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
    elif name == "OrganSMNIST":
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(severity=10, mixture_width=1, alpha=1.5),
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        Train_Data = medmnist.OrganSMNIST(size=size, split='train', download=False, root=Data_Path, transform=train_transform)
        Test_Data  = medmnist.OrganSMNIST(size=size, split='test', download=False, root=Data_Path, transform=valid_transform)
        # Test_Data = medmnist.OrganSMNIST(size=size, split='test', download=False, root=Data_Path + 'MEDMNIST',transform=valid_transform)
    else:
        raise ValueError("Wrong Name of Dataset")

    num_class  = len(Train_Data.info['label'])
    return Train_Data, Test_Data, num_class


class CRC5000Tiles(Dataset):
    """One deterministic, class-stratified partition of CRC-5000 tiles."""

    def __init__(self, root, split, seed, transform=None):
        if split not in {'train', 'valid', 'test'}:
            raise ValueError(f'Unsupported CRC-5000 split: {split}')
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.seed = int(seed)
        self.class_to_idx = {name: index for index, name in enumerate(CRC5000_CLASSES)}
        self.samples = self._make_samples()

    def _make_samples(self):
        if not self.root.is_dir():
            raise FileNotFoundError(
                f'CRC-5000 root was not found: {self.root}. '
                'Expected Kather_texture_2016_image_tiles_5000 under the project root.'
            )

        split_ranges = {
            'train': slice(0, 375),
            'valid': slice(375, 500),
            'test': slice(500, 625),
        }
        samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            files = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {'.tif', '.tiff'}
            ) if class_dir.is_dir() else []
            if len(files) != 625:
                raise RuntimeError(
                    f'CRC-5000 class {class_name} must contain 625 TIFF images, '
                    f'but found {len(files)} in {class_dir}.'
                )

            # Each class is shuffled independently so every split has the same
            # class balance (375/125/125 per class) and is reproducible by seed.
            shuffled = list(files)
            random.Random(self.seed + label).shuffle(shuffled)
            samples.extend((path, label) for path in shuffled[split_ranges[self.split]])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def CRC5000(size=150, purpose='train', seed=0, root=CRC5000_PATH):
    """Return CRC-5000 datasets with the paper-style 60/20/20 split.

    The original 5,000 150x150 RGB tiles are split independently in each of
    the eight classes: 375 train, 125 validation and 125 test images.  Search
    receives train followed by validation, allowing Search.py to use its 0.75
    train portion without exposing the held-out test set.  Final training uses
    the train partition only and evaluates on the test partition.
    """
    if purpose not in {'search', 'train'}:
        raise ValueError(f'Unsupported CRC-5000 purpose: {purpose}')

    # The referenced CRC-5000 protocol uses affine flips only.  Retain the
    # original 150x150 tile resolution; DIDS explicitly uses ceil-mode pooling
    # to keep its multi-scale branches aligned at later odd-size stages.
    # Normalization is shared across search, validation and final evaluation.
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_data = CRC5000Tiles(root, split='train', seed=seed, transform=train_transform)
    valid_data = CRC5000Tiles(root, split='valid', seed=seed, transform=eval_transform)
    test_data = CRC5000Tiles(root, split='test', seed=seed, transform=eval_transform)

    if purpose == 'search':
        return ConcatDataset((train_data, valid_data)), test_data, CRC5000_NUM_CLASSES
    return train_data, test_data, CRC5000_NUM_CLASSES


class LC25000Tiles(Dataset):
    """LC25000 tiles backed by a precomputed, fixed stratified split."""

    def __init__(self, root, samples, transform=None):
        self.root = Path(root)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample['path']
        with Image.open(image_path) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample['label'])


def _lc25000_image_files(class_dir):
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _create_lc25000_split(root, split_path):
    """Create the one-time, class-stratified 4000/500/500 split file."""
    splits = {'train': [], 'valid': [], 'test': []}
    source_counts = {}
    for label, class_name in enumerate(LC25000_CLASSES):
        class_dir = root / LC25000_CLASS_PATHS[class_name]
        if not class_dir.is_dir():
            raise FileNotFoundError(f'LC25000 class directory is missing: {class_dir}')
        files = _lc25000_image_files(class_dir)
        if len(files) != 5000:
            raise RuntimeError(
                f'Expected exactly 5000 images for LC25000 class {class_name}, found {len(files)}.'
            )
        # The split seed is intentionally independent of the training seed.
        # Once saved, every training run reuses exactly these image identities.
        rng = random.Random(LC25000_SPLIT_SEED + label)
        rng.shuffle(files)
        entries = [
            {'path': path.relative_to(root).as_posix(), 'label': label, 'class_name': class_name}
            for path in files
        ]
        splits['train'].extend(entries[:4000])
        splits['valid'].extend(entries[4000:4500])
        splits['test'].extend(entries[4500:5000])
        source_counts[class_name] = len(files)

    payload = {
        'dataset': 'LC25000',
        'version': 1,
        'split_seed': LC25000_SPLIT_SEED,
        'classes': list(LC25000_CLASSES),
        'source_counts': source_counts,
        'split_counts_per_class': {'train': 4000, 'valid': 500, 'test': 500},
        'splits': splits,
    }
    with split_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _load_lc25000_split(root, split_path):
    with split_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('dataset') != 'LC25000' or payload.get('classes') != list(LC25000_CLASSES):
        raise RuntimeError(f'LC25000 split file has incompatible metadata: {split_path}')
    expected_sizes = {'train': 20000, 'valid': 2500, 'test': 2500}
    for split_name, expected_size in expected_sizes.items():
        samples = payload.get('splits', {}).get(split_name)
        if not isinstance(samples, list) or len(samples) != expected_size:
            raise RuntimeError(
                f'LC25000 split {split_name!r} must contain {expected_size} samples in {split_path}.'
            )
        class_counts = [0] * LC25000_NUM_CLASSES
        for sample in samples:
            label = sample.get('label')
            relative_path = sample.get('path')
            if not isinstance(label, int) or not 0 <= label < LC25000_NUM_CLASSES:
                raise RuntimeError(f'Invalid LC25000 label in {split_path}: {sample}')
            if not isinstance(relative_path, str) or not (root / relative_path).is_file():
                raise RuntimeError(f'LC25000 split references a missing image: {relative_path}')
            class_counts[label] += 1
        per_class = {'train': 4000, 'valid': 500, 'test': 500}[split_name]
        if class_counts != [per_class] * LC25000_NUM_CLASSES:
            raise RuntimeError(
                f'LC25000 split {split_name!r} is not the required class-stratified partition: {class_counts}.'
            )
    return payload


def LC25000(size=224, root=LC25000_PATH, split_file=LC25000_SPLIT_FILE):
    """Return fixed-split LC25000 datasets for transfer-only retraining.

    The split is created once as 80/10/10 per class (4000/500/500) and is
    deliberately independent of training seeds.  Search.py does not use this
    dataset; Train.py uses it only for PathMNIST-to-LC25000 transfer training.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'LC25000 root directory does not exist: {root}')
    split_path = root / split_file
    if split_path.exists():
        payload = _load_lc25000_split(root, split_path)
    else:
        payload = _create_lc25000_split(root, split_path)

    # TM-NAS augmentation description: rotation, horizontal flip, scaling and
    # color dithering only.  Validation/test remain deterministic.
    mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    splits = payload['splits']
    return (
        LC25000Tiles(root, splits['train'], transform=train_transform),
        LC25000Tiles(root, splits['valid'], transform=eval_transform),
        LC25000Tiles(root, splits['test'], transform=eval_transform),
        LC25000_NUM_CLASSES,
        split_path,
    )
