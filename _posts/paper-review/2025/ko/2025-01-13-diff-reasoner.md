---
title: "[논문 리뷰] Are Diffusion Models Vision-And-Language Reasoners?"
author: invhun
date: 2025-01-13 15:00:00 +0900
last_modified_at: 2025-01-18 18:29:00 +0900
categories: [Paper Review, Multimodal Learning]
tags: [Diffusion-based, Multimodal Learning, Representation Learning, Text-Image, Retrieval, Text-Image Retrieval]
use_math: true
pin: false
lang: ko
---

> Are Diffusion Models Vision-And-Language Reasoners?  
> Neurips 2023   
> **Benno Krojer**, Elinor Poole-Dayan, Vikram Voleti, Christopher Pal, Siva Reddy   
> Mila University, McGill University, Polytechnique Montréal, Stability AI, ServiceNow Research, CIFAR AI   
> [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1a675d804f50509b8e21d0d3ca709d03-Paper-Conference.pdf)] [[github](https://github.com/McGill-NLP/diffusion-itm)]

# Abstract

Diffusion 기반 Text-conditioned 이미지 생성형 모델은 Discriminative한 Task에 적용하기 위해서 저자는 두 가지 혁신을 제안
1. Stable Diffusion을 변형하여, 이미지-텍스트 매칭(ITM) 태스크를 위한 새로운 방법 DiffusionITM을 제안.
2. 7개의 복잡한 비전-언어 작업, 편향 평가를 포함한 생성-판별 평가 벤치마크(GDBench)를 도입

제안 방식이 CLEVER와 Winoground와 같은 Compositionality 태스크에서 CLIP의 성능을 뛰어넘었으며, Stable Diffusion의 편항에 대해서 평가함

# 1. Introduction
- Text-Image 생성 기술이 빠르게 발전하고 있으며, 이들은 개방형 텍스트 포름프트의 Compositional 구조를 반영함
- Discriminative한 비전-언어 모델들은 판별에 필요한 최소한의 정보에만 집중하며, 이는 일반화되지 않는 임의의 상관관계일 수 있으며, 이를 평가하기 위해 Winoground, ARO와 같은 벤치마크가 제안됨
- 저자는 Compositinal 데이터를 합성하도록 학습된 생성 모델이, 이러한 복잡성을 해결할 수 있다고 가정하여 Stable Diffusion 기반 DiffusionITM 방식을 제안함
- Stable Diffusion(이하 SD)를 사용하여 입력된 이미지와 텍스트를 활용하여 가장 낮은 노이즈 예측 오류를 제공하는 텍스트 혹은 이미지를 선택하는 간단한 방식을 적용하였을 때, 이미지-텍스트에서는 우수한 성능을 보이나, 텍스트-이미지에서는 낮은 성능을 보임
- 이러한 현상을 분석하고, Hard negative로 fine-tuning하는 방식을 제안
- 7개의 복잡한 비전-언어 작업과, 사회적 편향 분석을 위한 편향 평가 데이터셋을 포함한 벤치마크 GDBench를 제안
- 실험결과 SD기반 DiffusionITM은 여러 태스크에서 CLIP과 경쟁력있는 결과를 보여주며, 어려운 Compositional 태스크에서는 CLIP을 초과함
- SD는 CLIP보다 편향이 낮으며, SD 2.1이 SD 1.5보다 편향이 적음을 관찰함

## 기존 연구 문제점:

- Discriminative vision-language 모델은 잘못된 상관관계에 의존하여 일반화 되지 않는 경우가 많음
- Diffusion을 단순 최소 노이즈 예측 방식으로 Retrieval에 사용할 때, ITR은 잘 작동하나, TIR은 시각적 속성에만 의존하여, 무작위 성능을 보임
- 생성 이미지 평가는 어렵고, 보조적인 모델의 사용이 필요함

## 제안 방법:

- DiffusionITM을 통해, diffusio모델을 zero-shot image-text 매칭에 맞게 변형
- MS-COCO에서 hard negative pair를 활용하여, fine-tuning방식 제안
- 다양한 이미지-텍스트 매칭과 bias 평가를 수행할 수 있는 GDBench 제안

# 2. Related Work
### Evaluation of Image Generation
- 전통적으로 이미지 품질과, 이미지-텍스트 정렬을 기반으로 평가를 수행함
- 기존 매트릭에는 FID, CLIPScore, Object detector 기반, 캡션+BLEU기반이 있음
- 최근 VQA모델을 사용하여, 생성된 이미지에 대한 질문에 답하도록 설계된 TIFA가 제안됨
- GDBench는 개별 예제에 대한 평가지표가 아닌, 전체적은 평가 프레임워크로, 다른 큰 모델이 평가에 필요하지 않음

### Bias in Image Generation Models
- 생성 모델에 대한 평가 기술의 부족으로 bias 평가는 discriminative 비전-언어 모델에 집중됨
- DALL-EVAL (ICCV2023): 백인과 남성에 연관된 집단에 편향되어 있음을 발견, SD 2.0이 1.4보다 편향되어 있음   
-> 하지만 평가에 시간이 오래걸리고 수동적임

# 3. Our Approach to Image-Text Matching with Diffusion Models
## 3.1. Diffusion Image-Text Matching: Overcoming the modality asymmetry for image retrieval
### Text-conditioned Diffusion
- 가정: Diffusion model에 입력된 이미지와 텍스트가 유사할 수록 특정 time-step t에서 원래 noise와 예측된 noise의 차이가 작을 것   
    -> 즉 식(1) Diffusion Loss의 값이 작을 것    
    ![fo1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRJmqkDs0b2PS8Hc?embed=1&width=700&height=82)

- Image2Text Retrieval
    - 각 이미지 별, 모든 텍스트 중 가장 낮은 diffusion loss 평균을 가지는 것을 선택
    - 각 이미지 별, Sampling한 time step들을 모든 텍스트에 대해서 동일하게 적용하여 평가   
    -> 우수한 성능을 보임   
    ![fo2.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRMxNrb6NU6pGf3P?embed=1&width=700&height=80)

- Text2Image Retrieval
    - Image2Text retrieval과 동일한 방식으로 평가
    - 하지만 random에 가까운 결과가 관찰됨   
    ![fo3.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRRYQbdEPVk_lxGx?embed=1&width=700&height=86)

- 문제점 분석
    ![fig3.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQ_wdK5YTMGWLgwa?embed=1&width=1624&height=552)
    - 이미지 기준: "이미지1"과 가장 유사한 "캡션1"이 가장 낮은 noise 예측 error를 보임
    - 텍스트 기준: "입력된 캡션"과 상관없이 "이미지2"의 error는 "이미지1"의 error보다 낮음
    - Diffusion model의 노이즈 제거는 주로 시각과 텍스트에 고르게 의존하지 않고, 시각적 속성에 의존함   
    -> 텍스트 조건과 상관없이 시각적으로 쉬운 이미지에 대해 가장 낮은 노이즈 예측 오류(Diffusion loss)를 가지게 됨

- 개선한 Text2Image Retrieval
    ![fig1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRdyxwxT2MY2sCWx?embed=1&width=1646&height=916)
    - 따라서 ("text conditioned error" – "un conditioned error") 가장 작은 경우를 선택하는 방식을 사용   
    ![fo5.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQrNe611PigZv9ah?embed=1&width=700&height=102)    

## 3.2. HardNeg-DiffusionITM: Tuning with compositional hard negatives and transfer
![fo6-7.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQ1D6-_7p_EZPWCD?embed=1&width=700&height=158)
- MS-COCO를 사용하여, 하드 네거티브에 대해 loss를 계산하여 fine-tuning함
- U-Net cross-attention에 LORA 레이어를 추가하고, 이를 fine-tuning하는 방식을 사용
- 모델이 positive prompt의 noise 예측 목표에서 너무 벗어나지 않도록 $L_{neg}$를 $[-L_{pos}, {L_{pos}}]$범위로 제한
- 이 Diffusion 기반 방식은 image과 text를 동시에 인코딩 하기 때문에 각각 인코딩 하는 방식인 CLIP과는 달리, 많은 negative sample을 batch에 포함할 수 없음   
-> MS-COCO로 fine-tuning 후 zero-shot방식으로 평가

# 4. Data: The GDBench Benchmark

- 목적: GDBench는 Diffusion 기반 생성 모델의 비전-언어 추론 작업에 대한 downstream task 성능을 측정하기 위한 벤치마크로 NLP의 GLUE 벤치마크와 유사한 방식으로 모델의 성능을 평가
- 구성: GDBench는 8개의 다양한 이미지-텍스트 매칭(ITM) 작업으로 구성되어 있으며, 7개는 ability 중심, 1개는 bias 데이터셋
- 장점: 명확하며, 다양성을 제공, 비전-언어 데이터셋에 대한 해석 가능한 평가를 제공함또한 VQA와 같은 별도의 모델 없이 평가를 수행
- 포함된 데이터셋:
    - Flickr30K: 다양한 장면의 이미지 및 텍스트 검색 데이터셋
    - Winoground: 조합 가능성을 평가하는 진단 벤치마크
    - ARO: 자동 생성된 데이터셋으로 하드 텍스트 네거티브 포함
    - ImageCoDe: 유사한 이미지와 복잡한 캡션에 초점
    - SVO: 주어, 목적어, 동사 분리에 따른 성능 평가
    - CLEVR2: 3D 형태의 이미지를 기반으로 다양한 현상 평가
    - Pets: 37종의 동물을 포함한 작은 이미지 분류 데이터셋

- ***기존 데이터셋들을 가져와서 통합한 벤치마크 제안이라고 볼 수 있음***

### Measuring Bias
![fo8.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRC4Yw5Zbn2IZeT0?embed=1&width=700&height=212)
- 세 가지 사회적 편향: Religious, Nationality, sexual orientation을 조사, 특정 집단과 속성 간의 연관성을 측정하여 평가함
- $\sigma$ 점수는 DiffusionITM의 점수 또는 CLIP의 경우 코사인 유사도
- 각 이미지에 대해 속성 A의 평균 점수에서 속성 B의 평균 점수를 뺌
- X집단에 대한 평균 점수와, Y집단에 대한 평균 점수를 각각 계산
- X와 Y의 모든 이미지에 대한 점수의 표준 편차를 계산
- 최종 양의 점수가 높을수록 X 집단에 긍정적인 속성(A)의 편향이 존재함을 뜻함
- 반대로 음의 점수가 높을수록 Y 집단에 긍정적인 속성(A)의 편향이 존재함을 뜻함

# 5. Experiments and Results
### Hyperparameters
- 샘플링:
    - 타임스텝 t는 [0, 1000] 범위에서 uniform 샘플링
    - table1 메인실험에선 250개의 샘플을 사용, 다른 실험에서는 10개의 샘플만 사용

- CLIP RN50x64와 OpenCLIP VIT-L/14와 공정한 비교를 수행

- MS-COCO fine-tuning: 배치사이즈 112, clipping 1.0, 8epoch 이후 체크포인트 선택

### Runtime
- Flickr30K image2text retrieval task에서 pair 당 10개의 샘플을 사용하여 평가 시, A6000 1대로 ***68***분 소요(OpenCLIP ViT-L/14는 4분 소요)
- Supplementary: CLIP RN50X64로 top-10선정 후 test를 수행   
-> 그럼에도 많은 inference 시간이 필요함

### DiffusionITM performance & HardNeg-DiffusionITM performance: 
![t1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRbB7kFTgLd4HJQx?embed=1&width=1634&height=1196)
- I2T는 일반적으로 CLIP보다 성능이 우수
- T2I는 단순한 Flickr가 아닌 어려운 데이터셋에서도 CLIP보다 낮은 성능을 보임

![t2.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQ55C1G7uzvpu624?embed=1&width=1630&height=1206)
- hard negative 유형에 따른 ablation study
    - 250샘플링 -> 10샘플링으로 줄였을 때, 큰 성능 하락이 있음
    - 단독으로 쓰일 때, hard neg는 random neg과 neg 없는것보다 성능이 떨어짐
    - Text retrieva에서는 negative의 영향이 적음

### Bias & Stable Diffusion 1.5 vs. 2.1 Performance:
![t3.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQt4Lu0vqOPHd1Pj?embed=1&width=1628&height=694)
![fig2.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgRGs6gygAIFUkYKd?embed=1&width=1634&height=672)
- CLIP과 stable diffusio은 기독교, 미국인, heterosexual에 편향된 경향이 있음
- 그 중 stable diffusion 2.1이 가장 낮은 편향을 보임
- 2.1의 safety filter의 약화가 다양성에 영향을 미쳤을 수 있음

# Review
- Diffusion model Discriminative Task인 ITR에 사용한 논문
- Diffusion model의 한번에 인코딩 하는 구조로 인해, Contrastive learning을 수행하기 어려움   
-> 이에 따라 많은 계산 복잡도에 비해, 낮은 성능을 기록
- Flickr, COCO가 아닌 ARO, Winoground, Sugarcrepe와 같은 어려운 벤치마크에서도 성능은 준수하나 너무 큰 계산복잡도로 인해 Diffusion으로 CLIP을 대체하기는 어려울 것 같음

