# SeMask FPN

> **SeMask: Semantically Masked Transformers for Semantic Segmentation**의 원본 repository는 [링크](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FPN)를 통해 확인할 수 있습니다.

<br>

## Contents

  - [Installation](#installation)
  - [Train](#train)
  - [Inference](#inference)
  - [Troubleshooting](#troubleshooting)
    - [Resolved Problems](#resolved-problems)
    - [Remaining Problems](#remaining-problems)
  - [Citing SeMask](#citing-semask)

<br>

## Installation

1. 기존의 `mmsegmentation`과 환경이 꼬이지 않도록 새로운 가상환경 `semask`를 생성합니다. 
   
   ```bash
   conda create -n semask python=3.7 -y
   
   source activate semask
   ```
2. 호환되는 버전의 `pytorch`, `torchvision`, `cudatoolkit`, `mmcv`를 설치합니다.
   
   ```bash
   conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch

   pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

   cd input/code/mmseg-SeMask-FPN # 코드가 위치한 경로로 이동

   pip install -e .  # or "python setup.py develop"
   ```
3. 앞으로 SeMask를 이용할 때는 해당 가상환경 하에서 이용합니다.

<br>

## Train

1. `configs` 디렉토리 내부에 자신의 이름으로 된 디렉토리를 만듭니다. (ex. `_snowman_`) 실험에 사용할 config 파일들은 해당 디렉토리 밑에 작성하면 됩니다.
2. `mmseg-SeMask-FPN/pretrain` 디렉토리를 생성하고, 그 밑에 원본 pretrained Swin-L 모델을 다운받습니다.
   
   ```bash
   mkdir pretrain

   wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
   ```
   > 기존의 `mmsegmentation`에서 pretrained Swin-L 모델을 사용할 때는 `tools/model_converters/swin2mmseg.py`를 이용하여 모델을 변환해야 했으나, `mmseg-SeMask-FPN`에서는 원본 모델을 사용하시면 됩니다. 
3. `mmsegmentation`과 동일하게 다음의 명령어로 학습을 수행할 수 있습니다.
   
   ```bash
   python tools/train.py [config-file-path] [--seed]
   ```

<br>

## Inference

1. jupyter lab의 커널에 `semask` 가상 환경을 추가합니다.
   
   ```bash
   python -m ipykernel install --user --display-name semask --name semask
   ```
2. jupyter lab에서 `inference.ipynb`를 열고, 1번에서 추가한 커널 `semask`로 변경합니다.
3. 차례대로 실행하면 됩니다. 이때, import 오류가 발생한다면 직접 `pip install` 해주시면 됩니다.

<br>

## Troubleshooting

### Resolved Problems
- [X] `Float`, `Half`와 관련된 type error
- [X] inference 시, dataset에 `shuffle=False`로 주었음에도 불구하고 shuffle이 되는 현상

### Remaining Problems
- [ ] 각 class에 대한 validation 결과가 `wandb`에 기록되지 않는 현상
- [ ] `save_best`가 동작하지 않는 현상

<br>

## Citing SeMask

```BibTeX
@article{jain2021semask,
  title={SeMask: Semantically Masking Transformer Backbones for Effective Semantic Segmentation},
  author={Jitesh Jain and Anukriti Singh and Nikita Orlov and Zilong Huang and Jiachen Li and Steven Walton and Humphrey Shi},
  journal={arXiv},
  year={2021}
}
```
