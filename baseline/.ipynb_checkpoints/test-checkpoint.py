from dataset import COCOSegDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    ToTensorV2()
])

train_dataset = COCOSegDataset('../data', '../data/train_all_new.json', mode='train', transform=train_transform)
print(len(train_dataset))
print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)
print(train_dataset[0][2])
