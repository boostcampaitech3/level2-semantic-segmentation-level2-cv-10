# [Lv2 P-Stage] Semantic Segmentation / #눈#사람
> 📑 Wrapup Report [▶︎ PDF](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/blob/master/Semantic%20Segmentation_CV_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(10%E1%84%8C%E1%85%A9).pdf)

## Members
| 김하준 | 송민수 | 심준교 | 유승리 | 이창진 | 전영우 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![눈사람_김하준](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![눈사람_송민수](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![눈사람_심준교](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![눈사람_유승리](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![눈사람_이창진](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![눈사람_전영우](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |
|[GitHub](https://github.com/HajunKim)|[GitHub](https://github.com/sooya233)|[GitHub](https://github.com/Shimjoonkyo)|[GitHub](https://github.com/seungriyou)|[GitHub](https://github.com/noisrucer)|[GitHub](https://github.com/wowo0709)|

<br>

## Competition : 재활용 품목 분류를 위한 Semantic Segmentation
<img width="1082" alt="image" src="https://user-images.githubusercontent.com/43572543/173501850-6134c00d-be1f-4a77-abd7-1ff9b9b8c96b.png">


### Introduction

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

- **Input** : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 COCO format으로 제공됩니다.
- **Output** : 모델은 pixel 좌표에 따라 카테고리 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다.

### Metric

Test set의 mIoU(Mean Intersection over Union)

<br>

## Overall Structure
![image](https://user-images.githubusercontent.com/43572543/173504154-5526fbf3-b0e1-4ecc-b804-0a16acbbbaaf.png)

<br>

## Main Contributions
```
- Multi-label Stratified Group K Fold
- Mislabeling 수정
- Lawin Transformer 추가
- SeMask-FPN 추가
- PyTorch Baseline 작성
- Pseudo Labeling
```

### PyTorch Baseline
[>> Baseline 폴더](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/baseline)

### MMSegmentation
[>> 개인별 config 폴더](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmsegmentation/configs)

[>> Lawin Transformer 추가](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmsegmentation/mmseg/models)

### SeMask-FPN
> 실행 방법은 `README.md` 참고

[>> mmseg-SeMask-FPN 폴더](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmseg-SeMask-FPN)

<br>


## Data
### EDA
<details>
<summary>category 별 개수 분포</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506897-6378bda4-92d5-4604-a0dc-209d3418b01a.png" width="500px" />
</details>

<details>
<summary>이미지 당 category 개수 분포</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506909-e8d37b8c-3d68-4bd1-bf63-13b71cd0926e.png" width="500px" />
</details>

<details>
<summary>category 별 area 분포</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506929-a6c06dfe-252f-437c-84a2-d67c92b66ab7.png" />
</details>


### Pre-Processing
#### Outlier Removal
- area가 1~30인 annotation을 제거하거나 mislabel 된 annotation(전반적인 minor 실수, 컵라면 컵 분류, 명함/전단지 분류, 비닐 봉투 분류)을 수정해보았으나 성능이 하락하였다.
- 그 원인은 다음과 같이 유추하였다.

  1. test set에서도 annotation에 대한 기준이 불명확했을 수 있다.
  2. 우리의 예상보다 더 정밀한 annotation guide line 이 존재하였으나 눈치채지 못했을 수 있다.

#### Multi-label Stratified Group K Fold
- 기존 train, validation set이 stratified group k fold 형태로 주어졌기 때문에 각 이미지 별 mask의 비중과 각 이미지 별 등장하는 category 수의 비율 또한 추가로 고려하기 위해서 multi-label stratified group k fold를 사용해보았다. 하지만 성능의 유의미한 변화가 없었다.
- 그 원인은 다음과 같이 유추하였다.

  1) 각 이미지의 annotation은 큰 마스크가 여러개로 쪼개진 경우가 많다. 이에 따라 하나의 물체임에도 여러 물체로 오인할 수 있다. 이에 따라 각 이미지에서 등장하는 동일 카테고리의 경우 한 카테고리 처리하였다. 이를 통해 multi-label 의 수를 최소화하는 효과도 얻을 수 있었으나 실제 이미 검수과정에서 확인 했듯이 각 이미지에는 동일한 카테고리의 물체가 여러번 등장하는 경우도 많기때문에 적절한 기준이 아니었을 수 있다. 대략적으로도 반영하는 것이 효과적일 것이라 생각하였으나, 이에 대한 근거도 충분하지 않다.
  2) 각 이미지의 mask 크기의 비율을 맞춰주는 것이 정말 성능이 영향이 있는지에 대한 선행 확인이 없었다. 결국에는 실험을 통해 확인할 수 밖에 없는 영역이었고 이번 구현을 통해 확인했다고 생각할 수 있다. (물론 category 수를 같이 반영했다는 점에서 영향이 있을 수 있으나, category 수, 이미지 마스크 크기를 랜덤으로 나눈 것과 성능이 비슷하다는 점에서 유의미하지 않다고 볼 수 있다.)

<br>

## Experiments
### Model
> MMSegmentation에서 제공하지 않는 Lawin과 SeMask를 추가하여 실험을 수행하였다.

| Backbone  |  Neck            |  Optimizer        |  Scheduler  |  특이사항               |
|-----------|------------------|-------------------|-------------|-------------------------|
| Lawin     |  Swin-L           |  -              |  AdamW      |  CosineAnnealing        |
| UperNet   |  Swin-L          |  -                |  AdamW      |  Poly, CosineRestart    |
| SeMask    |  SeMask Swin-L   |  FPN              |  AdamW      |  Poly, CosineRestart    |
| BEiT      |  BEiT            |  feature2pyramid  |  AdamW      |  CosineAnnealing        |
| K-Net      |  ResNet, Swin-L  |  -                |  AdamW      |  Poly, CosineAnnealing  |

### Pseudo-Labeling
각 모델 별 가장 성능이 좋았던 모델로 test셋 이미지에 pseudo labeling을 진행하였고, 유의미한 성능 향상을 확인 할 수 있었다. 다만, ensemble 한 결과, 즉 가장 성능이 좋은 결과로 pseudo labeling을 진행 하였으면 성능 향상이 어땠을지에 대한 의문이 여전히 남아있다. 다만, 이는 각 모델의 다양성이 줄어드는 결과를 가져왔을 염려가 있다.

### TTA
Test time 에서 scale의 ratio를 `[0.75, 1.0, 1.25, 1.5]`로 지정하여 multi scale augmentation을 수행한 결과, LB score 기준 0.01의 성능 향상을 확인할 수 있었다.

### Ensemble
최종적으로, 좋은 결과가 나왔던 모델들을 모아 앙상블을 시도하였다. 이때, hard voting 방법으로 앙상블을 진행하였는데, 각각의 pixel에서 model들이 예측한 class중 가장 많이 예측한 class를 이용하되 동점이라면 성능이 좋았던 모델부터 예측한 class가 예측한 후보중에 있다면 이를 result로 선택하는 방법을 택하였다. 이와 같은 같은 앙상블 기법을 도입한 결과, 최종적으로 LB score 0.7901을 달성할 수 있었다.

<br>

## LB Score Chart
<img src="https://user-images.githubusercontent.com/43572543/173503334-636e795f-dd1b-4f57-9efb-cf02031fd388.png" width="600px"/>

### Final LB Score
- **[Public]** mIoU: 0.7901
- **[Private]** mIoU: 0.7230
