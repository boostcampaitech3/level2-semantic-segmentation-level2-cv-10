import torch
from torch.utils.data import DataLoader
from data_loader.dataset import COCOSegDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(mode='train'):
    if mode == 'train':
        transform = A.Compose([
        A.HorizontalFlip(0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
        ])

    elif mode == 'val':
        transform = A.Compose([
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

    return transform

def collate_fn(batch):
    return tuple(zip(*batch))


class COCOSegDataLoader(DataLoader):
    def __init__(self, data_dir, json_path, mode='train', batch_size=16, shuffle=True, drop_last=True, num_workers=2):
        transform = get_transform(mode=mode)
        dataset = COCOSegDataset(data_dir, json_path, mode, transform=transform)

        init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'drop_last': drop_last,
            'num_workers': num_workers
        }

        super().__init__(**init_kwargs)
