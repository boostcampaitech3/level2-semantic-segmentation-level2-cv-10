# [Lv2 P-Stage] Semantic Segmentation / #๋#์ฌ๋
> ๐ Wrapup Report [โถ๏ธ PDF](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/blob/master/Semantic%20Segmentation_CV_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(10%E1%84%8C%E1%85%A9).pdf)

## Members
| ๊นํ์ค | ์ก๋ฏผ์ | ์ฌ์ค๊ต | ์ ์น๋ฆฌ | ์ด์ฐฝ์ง | ์ ์์ฐ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![แแฎแซแแกแแกแท_แแตแทแแกแแฎแซ](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![แแฎแซแแกแแกแท_แแฉแผแแตแซแแฎ](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![แแฎแซแแกแแกแท_แแตแทแแฎแซแแญ](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![แแฎแซแแกแแกแท_แแฒแแณแผแแต](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![แแฎแซแแกแแกแท_แแตแแกแผแแตแซ](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![แแฎแซแแกแแกแท_แแฅแซแแงแผแแฎ](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |
|[GitHub](https://github.com/HajunKim)|[GitHub](https://github.com/sooya233)|[GitHub](https://github.com/Shimjoonkyo)|[GitHub](https://github.com/seungriyou)|[GitHub](https://github.com/noisrucer)|[GitHub](https://github.com/wowo0709)|

<br>

## Competition : ์ฌํ์ฉ ํ๋ชฉ ๋ถ๋ฅ๋ฅผ ์ํ Semantic Segmentation
<img width="1082" alt="image" src="https://user-images.githubusercontent.com/43572543/173501850-6134c00d-be1f-4a77-abd7-1ff9b9b8c96b.png">


### Introduction

๋ฐ์ผํ๋ก ๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ , ์๋น๋๋ ์๋๋ฅผ ์ด๊ณ  ์์ต๋๋ค. ํ์ง๋ง ์ด๋ฌํ ๋ฌธํ๋ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์์ต๋๋ค.

๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค.

๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Detection ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก๋ ์ผ๋ฐ ์ฐ๋ ๊ธฐ, ํ๋ผ์คํฑ, ์ข์ด, ์ ๋ฆฌ ๋ฑ 10 ์ข๋ฅ์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋ฉ๋๋ค.

์ฌ๋ฌ๋ถ์ ์ํด ๋ง๋ค์ด์ง ์ฐ์ํ ์ฑ๋ฅ์ ๋ชจ๋ธ์ ์ฐ๋ ๊ธฐ์ฅ์ ์ค์น๋์ด ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ๋๊ฑฐ๋, ์ด๋ฆฐ์์ด๋ค์ ๋ถ๋ฆฌ์๊ฑฐ ๊ต์ก ๋ฑ์ ์ฌ์ฉ๋  ์ ์์ ๊ฒ์๋๋ค. ๋ถ๋ ์ง๊ตฌ๋ฅผ ์๊ธฐ๋ก๋ถํฐ ๊ตฌํด์ฃผ์ธ์! ๐

- **Input** :ย ์ฐ๋ ๊ธฐ ๊ฐ์ฒด๊ฐ ๋ด๊ธด ์ด๋ฏธ์ง๊ฐ ๋ชจ๋ธ์ ์ธํ์ผ๋ก ์ฌ์ฉ๋ฉ๋๋ค. segmentation annotation์ COCO format์ผ๋ก ์ ๊ณต๋ฉ๋๋ค.
- **Output** :ย ๋ชจ๋ธ์ย pixel ์ขํ์ ๋ฐ๋ผ ์นดํ๊ณ ๋ฆฌ ๊ฐ์ ๋ฆฌํดํฉ๋๋ค. ์ด๋ฅผ submission ์์์ ๋ง๊ฒ csv ํ์ผ์ ๋ง๋ค์ด ์ ์ถํฉ๋๋ค.

### Metric

Test set์ mIoU(Mean Intersection over Union)

<br>

## Overall Structure
![image](https://user-images.githubusercontent.com/43572543/173504154-5526fbf3-b0e1-4ecc-b804-0a16acbbbaaf.png)

<br>

## Main Contributions
```
- Multi-label Stratified Group K Fold
- Mislabeling ์์ 
- Lawin Transformer ์ถ๊ฐ
- SeMask-FPN ์ถ๊ฐ
- PyTorch Baseline ์์ฑ
- Pseudo Labeling
```

### PyTorch Baseline
[>> Baseline ํด๋](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/baseline)

### MMSegmentation
[>> ๊ฐ์ธ๋ณ config ํด๋](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmsegmentation/configs)

[>> Lawin Transformer ์ถ๊ฐ](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmsegmentation/mmseg/models)

### SeMask-FPN
> ์คํ ๋ฐฉ๋ฒ์ `README.md` ์ฐธ๊ณ 

[>> mmseg-SeMask-FPN ํด๋](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-10/tree/master/mmseg-SeMask-FPN)

<br>


## Data
### EDA
<details>
<summary>category ๋ณ ๊ฐ์ ๋ถํฌ</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506897-6378bda4-92d5-4604-a0dc-209d3418b01a.png" width="500px" />
</details>

<details>
<summary>์ด๋ฏธ์ง ๋น category ๊ฐ์ ๋ถํฌ</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506909-e8d37b8c-3d68-4bd1-bf63-13b71cd0926e.png" width="500px" />
</details>

<details>
<summary>category ๋ณ area ๋ถํฌ</summary>
<img src="https://user-images.githubusercontent.com/43572543/173506929-a6c06dfe-252f-437c-84a2-d67c92b66ab7.png" />
</details>


### Pre-Processing
#### Outlier Removal
- area๊ฐ 1~30์ธ annotation์ ์ ๊ฑฐํ๊ฑฐ๋ mislabel ๋ annotation(์ ๋ฐ์ ์ธ minor ์ค์, ์ปต๋ผ๋ฉด ์ปต ๋ถ๋ฅ, ๋ชํจ/์ ๋จ์ง ๋ถ๋ฅ, ๋น๋ ๋ดํฌ ๋ถ๋ฅ)์ ์์ ํด๋ณด์์ผ๋ ์ฑ๋ฅ์ด ํ๋ฝํ์๋ค.
- ๊ทธ ์์ธ์ ๋ค์๊ณผ ๊ฐ์ด ์ ์ถํ์๋ค.

  1. test set์์๋ annotation์ ๋ํ ๊ธฐ์ค์ด ๋ถ๋ชํํ์ ์ ์๋ค.
  2. ์ฐ๋ฆฌ์ ์์๋ณด๋ค ๋ ์ ๋ฐํ annotation guide line ์ด ์กด์ฌํ์์ผ๋ ๋์น์ฑ์ง ๋ชปํ์ ์ ์๋ค.

#### Multi-label Stratified Group K Fold
- ๊ธฐ์กด train, validation set์ด stratified group k fold ํํ๋ก ์ฃผ์ด์ก๊ธฐ ๋๋ฌธ์ ๊ฐ ์ด๋ฏธ์ง ๋ณ mask์ ๋น์ค๊ณผ ๊ฐ ์ด๋ฏธ์ง ๋ณ ๋ฑ์ฅํ๋ category ์์ ๋น์จ ๋ํ ์ถ๊ฐ๋ก ๊ณ ๋ คํ๊ธฐ ์ํด์ multi-label stratified group k fold๋ฅผ ์ฌ์ฉํด๋ณด์๋ค. ํ์ง๋ง ์ฑ๋ฅ์ ์ ์๋ฏธํ ๋ณํ๊ฐ ์์๋ค.
- ๊ทธ ์์ธ์ ๋ค์๊ณผ ๊ฐ์ด ์ ์ถํ์๋ค.

  1) ๊ฐ ์ด๋ฏธ์ง์ annotation์ ํฐ ๋ง์คํฌ๊ฐ ์ฌ๋ฌ๊ฐ๋ก ์ชผ๊ฐ์ง ๊ฒฝ์ฐ๊ฐ ๋ง๋ค. ์ด์ ๋ฐ๋ผ ํ๋์ ๋ฌผ์ฒด์์๋ ์ฌ๋ฌ ๋ฌผ์ฒด๋ก ์ค์ธํ  ์ ์๋ค. ์ด์ ๋ฐ๋ผ ๊ฐ ์ด๋ฏธ์ง์์ ๋ฑ์ฅํ๋ ๋์ผ ์นดํ๊ณ ๋ฆฌ์ ๊ฒฝ์ฐ ํ ์นดํ๊ณ ๋ฆฌ ์ฒ๋ฆฌํ์๋ค. ์ด๋ฅผ ํตํด multi-label ์ ์๋ฅผ ์ต์ํํ๋ ํจ๊ณผ๋ ์ป์ ์ ์์์ผ๋ ์ค์  ์ด๋ฏธ ๊ฒ์๊ณผ์ ์์ ํ์ธ ํ๋ฏ์ด ๊ฐ ์ด๋ฏธ์ง์๋ ๋์ผํ ์นดํ๊ณ ๋ฆฌ์ ๋ฌผ์ฒด๊ฐ ์ฌ๋ฌ๋ฒ ๋ฑ์ฅํ๋ ๊ฒฝ์ฐ๋ ๋ง๊ธฐ๋๋ฌธ์ ์ ์ ํ ๊ธฐ์ค์ด ์๋์์ ์ ์๋ค. ๋๋ต์ ์ผ๋ก๋ ๋ฐ์ํ๋ ๊ฒ์ด ํจ๊ณผ์ ์ผ ๊ฒ์ด๋ผ ์๊ฐํ์์ผ๋, ์ด์ ๋ํ ๊ทผ๊ฑฐ๋ ์ถฉ๋ถํ์ง ์๋ค.
  2) ๊ฐ ์ด๋ฏธ์ง์ mask ํฌ๊ธฐ์ ๋น์จ์ ๋ง์ถฐ์ฃผ๋ ๊ฒ์ด ์ ๋ง ์ฑ๋ฅ์ด ์ํฅ์ด ์๋์ง์ ๋ํ ์ ํ ํ์ธ์ด ์์๋ค. ๊ฒฐ๊ตญ์๋ ์คํ์ ํตํด ํ์ธํ  ์ ๋ฐ์ ์๋ ์์ญ์ด์๊ณ  ์ด๋ฒ ๊ตฌํ์ ํตํด ํ์ธํ๋ค๊ณ  ์๊ฐํ  ์ ์๋ค. (๋ฌผ๋ก  category ์๋ฅผ ๊ฐ์ด ๋ฐ์ํ๋ค๋ ์ ์์ ์ํฅ์ด ์์ ์ ์์ผ๋, category ์, ์ด๋ฏธ์ง ๋ง์คํฌ ํฌ๊ธฐ๋ฅผ ๋๋ค์ผ๋ก ๋๋ ๊ฒ๊ณผ ์ฑ๋ฅ์ด ๋น์ทํ๋ค๋ ์ ์์ ์ ์๋ฏธํ์ง ์๋ค๊ณ  ๋ณผ ์ ์๋ค.)

<br>

## Experiments
### Model
> MMSegmentation์์ ์ ๊ณตํ์ง ์๋ Lawin๊ณผ SeMask๋ฅผ ์ถ๊ฐํ์ฌ ์คํ์ ์ํํ์๋ค.

| Backbone  |  Neck            |  Optimizer        |  Scheduler  |  ํน์ด์ฌํญ               |
|-----------|------------------|-------------------|-------------|-------------------------|
| Lawin     |  Swin-L           |  -              |  AdamW      |  CosineAnnealing        |
| UperNet   |  Swin-L          |  -                |  AdamW      |  Poly, CosineRestart    |
| SeMask    |  SeMask Swin-L   |  FPN              |  AdamW      |  Poly, CosineRestart    |
| BEiT      |  BEiT            |  feature2pyramid  |  AdamW      |  CosineAnnealing        |
| K-Net      |  ResNet, Swin-L  |  -                |  AdamW      |  Poly, CosineAnnealing  |

### Pseudo-Labeling
๊ฐ ๋ชจ๋ธ ๋ณ ๊ฐ์ฅ ์ฑ๋ฅ์ด ์ข์๋ ๋ชจ๋ธ๋ก test์ ์ด๋ฏธ์ง์ pseudo labeling์ ์งํํ์๊ณ , ์ ์๋ฏธํ ์ฑ๋ฅ ํฅ์์ ํ์ธ ํ  ์ ์์๋ค. ๋ค๋ง, ensemble ํ ๊ฒฐ๊ณผ, ์ฆ ๊ฐ์ฅ ์ฑ๋ฅ์ด ์ข์ ๊ฒฐ๊ณผ๋ก pseudo labeling์ ์งํ ํ์์ผ๋ฉด ์ฑ๋ฅ ํฅ์์ด ์ด๋ ์์ง์ ๋ํ ์๋ฌธ์ด ์ฌ์ ํ ๋จ์์๋ค. ๋ค๋ง, ์ด๋ ๊ฐ ๋ชจ๋ธ์ ๋ค์์ฑ์ด ์ค์ด๋๋ ๊ฒฐ๊ณผ๋ฅผ ๊ฐ์ ธ์์ ์ผ๋ ค๊ฐ ์๋ค.

### TTA
Test time ์์ scale์ ratio๋ฅผ `[0.75, 1.0, 1.25, 1.5]`๋ก ์ง์ ํ์ฌ multi scale augmentation์ ์ํํ ๊ฒฐ๊ณผ, LB score ๊ธฐ์ค 0.01์ ์ฑ๋ฅ ํฅ์์ ํ์ธํ  ์ ์์๋ค.

### Ensemble
์ต์ข์ ์ผ๋ก, ์ข์ ๊ฒฐ๊ณผ๊ฐ ๋์๋ ๋ชจ๋ธ๋ค์ ๋ชจ์ ์์๋ธ์ ์๋ํ์๋ค. ์ด๋, hard voting ๋ฐฉ๋ฒ์ผ๋ก ์์๋ธ์ ์งํํ์๋๋ฐ, ๊ฐ๊ฐ์ pixel์์ model๋ค์ด ์์ธกํ class์ค ๊ฐ์ฅ ๋ง์ด ์์ธกํ class๋ฅผ ์ด์ฉํ๋ ๋์ ์ด๋ผ๋ฉด ์ฑ๋ฅ์ด ์ข์๋ ๋ชจ๋ธ๋ถํฐ ์์ธกํ class๊ฐ ์์ธกํ ํ๋ณด์ค์ ์๋ค๋ฉด ์ด๋ฅผ result๋ก ์ ํํ๋ ๋ฐฉ๋ฒ์ ํํ์๋ค. ์ด์ ๊ฐ์ ๊ฐ์ ์์๋ธ ๊ธฐ๋ฒ์ ๋์ํ ๊ฒฐ๊ณผ, ์ต์ข์ ์ผ๋ก LB score 0.7901์ ๋ฌ์ฑํ  ์ ์์๋ค.

<br>

## LB Score Chart
<img src="https://user-images.githubusercontent.com/43572543/173503334-636e795f-dd1b-4f57-9efb-cf02031fd388.png" width="600px"/>

### Final LB Score
- **[Public]** mIoU: 0.7901
- **[Private]** mIoU: 0.7230
