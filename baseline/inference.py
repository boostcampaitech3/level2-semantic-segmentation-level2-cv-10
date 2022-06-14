import torch
import albumentations as A
import segmentation_models_pytorch as smp
from utils import read_json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from data_loader import COCOSegDataLoader
from models import build_smp_model


def inference(model, data_loader, device):
    model.eval()
    size=256
    transform = A.Compose([A.Resize(size, size)])

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for imgs, img_infos in tqdm(test_loader):
            imgs = torch.stack(imgs, dim=0) # (16, 3, 512, 512)

            outputs = model(imgs.to(device))
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy() # (16, 512, 152)

            temp_mask = []
            for img, mask in zip(imgs.numpy(), outputs):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask'] # (256, 256)
                temp_mask.append(mask)

            oms = np.array(temp_mask) #(16, 256, 256)
            oms = oms.reshape(oms.shape[0], size*size).astype(int) # (16, 256*256)
            preds_array = np.vstack([preds_array, oms]) # (16,256*256)->(32,256*256)->...

            file_name_list.extend([info['file_name'] for info in img_infos])

        print("End prediction")

        return file_name_list, preds_array


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_config = read_json('test_config.json')

    # Load checkpoint
    checkpoint = torch.load(test_config['model_path'], map_location=device)

    # Initialize Model
    model_config = checkpoint['model_config']
    model = build_smp_model(model_config)

    # Load Model state dict
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    model = model.to(device)

    # Test Loader
    test_loader = COCOSegDataLoader(**test_config['test_loader']['args'])

    # Inference
    fnames, preds = inference(model, test_loader, device)

    # Submission
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    for fname, pred_1d in zip(fnames, preds):
        submission = submission.append({
            "image_id": fname,
            "PredictionString": ' '.join([str(e) for e in pred_1d.tolist()])
        }, ignore_index=True)

    submission.to_csv(test_config['submission_path'], index=False)


