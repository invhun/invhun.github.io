---
title: "[논문 리뷰] Unified Lexical Representation for Interpretable Visual-Language Alignment"
author: invhun
date: 2025-02-04 14:00:00 +0900
last_modified_at: 2025-02-04 14:00:00 +0900
categories: [Paper Review, Multimodal Learning]
tags: [Text-Image, Text-Image Retrieval, Multimodal Learning, DINOv2, Overcoming CLIP Limitations, Zero-shot Retrieval, Lexical Representation, Representation Learning]
use_math: true
pin: false
lang: ko
---

> Unified Lexical Representation for Interpretable Visual-Language Alignment  
> NeurIPS 2024 Poster   
> **Yifan Li**, Yikai Wang, Yanwei Fu, Dongyu Ru, Zheng Zhang, Tong He    
> Fudan University, Amazon Web Services    
> [[paper](https://arxiv.org/pdf/2407.17827)] [[Project page](https://clementine24.github.io/LexVLA/)] [[github](https://github.com/Clementine24/LexVLA)]

# 1. Abstract & Introduction

### 기존 연구 문제점
- CLIP의 잠재 특징은 해석 가능성이 부족하여 개별 요소의 영향을 분석하기 어려움
- CLIP의 시각 모델은 패치 수준 특징을 잘 학습하지 못하고, 텍스트 모델은 불완전하고 편향된 캡션으로 학습됨
- 어휘 표현은 명확하지만 CLIP보다 훨씬 큰 임베딩 차원, 부족한 supervised signal의 이유로 학습이 어려움 -> CLIP 스타일의 VLA모델에서 어휘 표현을 사용하기는 어려움
- 기존 어휘 표현 학습 방식은, 복잡한 구성을 요구하며, 학습의 어려움을 증가

### 제안 방법
- LexVLA는 CLIP 스타일의 대조 학습 파이프라인에서 통합 어휘 표현을 학습하는 간단하면서 포괄적인 프레임워크임
- DINOv2와 Llama 2를 사용하여 각각 비전 모델과 텍스트 모델의 강점을 활용.
통합 어휘를 위한 코드북을 사용하여 단일 모달 모델의 특성을 유지하고, 적은 다중 모달 훈련 데이터로 효과적인 VLA 모델을 구성
- Overuse 패널티를 도입하여 무의미한 토큰의 과도한 활성화를 방지하고, 패치 수준 평가를 위한 PatchDis 메트릭을 제안


# 2. Related works
### Vision-Language Alignment
- CLIP의 혁신으로 많은 연구가 진행되었고, 최근 LLMs의 발전으로 CLIP과 LLM을 결합하는 것이 일반화 되었음
- 이러한 방법에서, 시각 인코더의 병목현상과 속성 무시, 할루시네이션 등의 이슈가 제기됨
- CLIP대신 다른 모델을 사용하는 연구의 발전은 더뎠고, 많은 양의 코스트를 필요로함
- 저자의 방식은 고정된 DINOv2를 사용하여, 적은 양의 멀티모달 데이터로 CLIP보다 우수한 성능을 달성

### Multi-modal lexical representation
- Lexical representation은 해석가능성과 효율성으로 information retrieval에서 많이 사용됨
- 비전-언어 모델에 적용한 기존 연구는 복잡한 학습 단계(2 or 3 stage), 대규모 학습 데이터, Bag-of-words(BoW)제한을 필요로 함
- 저자는 단일 단계 fine-tuning으로 unimodal pretrained 모델을 어휘 공간에 정렬하는 방식을 제안, 이는 적은 데이터와 BoW없이 가능함

# LexVLA
![fig2.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQRVL2aLLQqjQbZzBEkFCJkNAdd0nRRu2BeXTmJrHFS1y2g?width=1024)
### problem setup
- 일반적으로 어휘 표현 $s_i \in \mathbb{R}^V, i \in \lbrace img, txt \rbrace$ 는 점수벡터로, 각 요소는 샘플과 어휘 $V$의 해당 단어 간의 유사성을 나타냄.
- 대조 어휘 정렬은 각 모달리티에 대해 어휘 인코더를 학습하고, 긍정 이미지-텍스트 쌍의 유사성을 최대화, 부정 쌍의 유사성을 최소화하는 것이 목표

## 3.1. Lexical representation
### Vocabulary
- 기존 토크나이제이션 기법에 따라, 토크나이저의 어휘로 초기화하고, 의미없는 토큰을 제거하여 32,000->17,149로 축소

### Codebooks
- one-hot 임베딩으로 구성하지 않고, 4096차원의 벡터를 구성
- 동일한 어휘를 사용하되, 모달리티 별 고유한 코드북을 구성. 이때 텍스크 코드북 $Z_{txt}$는 frozen하고, 이미지 코드북 $Z_{img}$는 $Z_{txt}$로 초기화 후 fine-tuning함

### Sparse representation
- dense output vector -> sparse lexical representation으로 변환은 자연스럽게 가능함
- 저자는 여러 방법 중, 임계값($1/\sqrt{V}$) 이상의 값을 가진 항목만 유지하는 방식을 선택

## 3.2. Lexical encoder

![fo1.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQQLdDcdiQ_zRqomQiw1ZB6HAUKmweAmgAFlvvHI4Sa_HOU?width=1024)
1. 단일 모달 특징 추출
2. lexical feature sequence로 projection
3. lexical representation으로 변환

### 3.2.1. Lexical text encoder

#### Captioning?
- VLA 훈련 데이터에서 텍스트는 일반적으로 해당 이미지의 캡션임.
하지만 캡션은 이미지의 의미를 완벽하게 포착하지 못하며, 이전 접근 방식은 편향된 정렬을 초래함.

#### Predicting!
- 강력한 LLM의 내재 지식을 활용하는 방안 탐구

![prompt1.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTxvS1pnwrmR5DgAItZWeg5ASdGcD6ZTQSwOr-dde-Oqhk?width=1024)

- 위 프롬프트를 사용하여, prediction 및 중요 단어를 식별함

#### Realization

![fo2.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQT5tUr9gIcaTbIf_EWEVeSRAbngVRKSbF4o1z_DuX3h-S4?width=1024)

- Llama2를 텍스트 인코더로 활용

### 3.2.2. Lexical visual encoder

![fo3.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTBUBDCWhZ5T7YYegmx3_ExAZyH52XKLcdPc3_TcLldkWA?width=1024)
![fo4.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQRd9iQUXVtTSIrOB7GgFSs-AYAvO7bH4JZEwgzFgssqfXI?width=1024)

- DINOv2 사용: DINOv2를 비주얼 백본으로 활용하여 이미지 인코딩.
- 어댑터: self-attention, MLP로 구성
- 맵퍼: 이미지 패치 토큰과 어휘 코드북 간의 dot product 계산 후 ELU1p activation 적용
- max 풀링 및 정규화: 패치 표현을 집계하여 글로벌 어휘 표현을 생성.
- 패치 표현: 이미지 패치 어휘 표현은 유사한 과정을 따르지만, 최대 풀링 대신 패치 위치 선택.

## 3.3. Train LexVLA

#### Contrastive objective
![fo5.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTafFMHCz_1QrhW3X78tDWaAaJHdRn3BCOI7vsWtcIwAcQ?width=1024)
- 일반적으로 사용되는 InfoNCE 로스를 사용

#### overuse penalty
![fo6.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQQgl2t6gkYiSqUNVqbEn-PpAdqMab51BpE6f56WHNcxzuk?width=1024)
![fo7.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTnuJ4VThfbQo6mVTGKQPRRAUs3D6UmWGjmOowDNtDmteg?width=1024)

- FLOPs 손실은 sparsity를 유도하기 위해, FLOPs를 줄이는 것을 목표로 함. 하지만 이 방식이 관련 없는 토큰을 잘못 활성화하도록 유도하는 short cut을 취하게 만드는 경우가 있음

- 이를 방지하기 위해, 너무 자주 사용되는 토큰에 패널티를 부여하는 항을 추가함

#### 최종 로스
![fo8.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQQM7tmbdDJ3RYoOFnS_x83RAevRS07V0rZ6tKrgytYNEHI?width=1024)


#### incremental fine-tuning
Llama2는 LoRA로 fine-tuning
DINOv2는 frozen 하고, 프로젝터와 vision codebook를 학습


## 3.4. PatchDis: interpretability metric
- PatchDis: 패치 수준의 비주얼 어휘 표현 해석 가능성을 평가하기 위한 메트릭.
- 패치 수준 분류: 텍스트 인코더를 통해 클래스 임베딩을 얻고, 비주얼 인코더를 통해 패치 특징을 추출하여 유사도 계산.
- 집계 및 평가: 클래스별로 패치를 집계하고, 정답 세분화와의 비교를 통해 mIoU를 사용하여 해석 가능성을 평가.
- *sementic segmentation이랑 비슷한 방식인듯 하다*

![fig3.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQT1YQ9G2uAgSrIbtdAtu9JRAVq3C4Cd8-OpK77XUOzdMn8?width=1024)

# 4. Experiments

#### Implementation details

![t3.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTB73WqChYQQbp93H-2BwUcAWR-B7F2I8tSXnBZrWc3m2g?width=1024)
- 40GB A100 8대를 사용하였음
- 학습가능한 파라미터는 Vision codebook에 70M이 있고, vision projector에 17M, LLaMa LoRa에 21M로 구성됨. 코드북에 생각보다 많음
- Neurips는 보통 supplementary에 매우 자세한 Hyperparameter를 기록하는 것 같음
- Datasets & Benchmark track에 투고할 때 나도 저렇게 구성했었다. 리젝됐지만...

## 4.1. Zero-shot cross-modal retrieval

![t1.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQR9GvjYLioySoGOxUaO95DSAYpe5TT2jGfwKgr8XNdrOqs?width=1024)
- CC-12M 중 다운로드한 9.2M개의 데이터를 학습에 사용
- Flickr, MSCOCO로 zero-shot 평가를 수행
- BoW: LLM기반 대신, 캡션에서 명사, 형용사, 동사만 선택하는 방식
- CLIP: DINOv2대신 CLIP ViT 사용하는 방식
- FLOPs: overuse penalty 제거
- 512: CLIP과 공정한 비교를 위해 CLIP과 같은 차원으로 설정, Top-k 방식으로 활성화 사용(LexVLA의 평군 활성화 차원은 1081)


![fig6.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQR2LXK5pMC_Q5z5xXIy_Ys9ARMhGngec4c3Qscg9rFPvfg?width=1024)
- overuse penalty를 제거하고 FLOPs 페널티만 사용하였을 때, 의미없는 토큰이 활성화 되는 것을 볼 수 있음

## 4.2. Lexical representation analysis


![t2.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQSti1anOJ7DQZ9qyk-0pqVrASeLl6mLfBSGnvSh8m8Gxi0?width=1024)
![fig4.PNG](https://1drv.ms/i/c/af5642aec05791fb/IQTiaw2G5gn8SKtQk4mTH3APAVQtCOT_3yDBZTDDRN7OEN4?width=1024)

- 저자가 제안한 평가지표로, CLIP, VDR과 비교를 수행함. CLIP의 낮은 해석 가능성과, 그 반대의 성능을 보여주는 LexVLA
- LexVLA의 DINOv2를 CLIP으로 바꿔도, 어느정도의 해석 가능성 향상이 존재함 -> 제안 프레임워크 자체의 우수성
- Fig4는 로컬 패치의 어휘 표현을 시각화함. 또한 더 큰 단어는 더 큰 어휘값을 나타내는 것을 관찰 할 수 있음


# Review
- CLIP의 문제점 지적과 기존 멀티모달 분야에서의 Lexical representation 방식의 문제점을 잘 극복한 논문
- 최근에 CLIP 방식의 문제점(로컬 특징 미약, 편향되고 짧은 캡션)을 지적하는 논문이 많이 나오고 있음
- CLIP의 문제점을 지적하면서 크게 4가지의 방식의 연구가 진행되고 있고, 생각보다 많이 나오는 것 같음. 하나의 모델에 대한 문제점을 해결하는게 4년이 지났음에도 여전히 나온다는게 신기하게 느껴짐
    1.CLIP을 frozen하고, 일부분을 수정해서 사용함
    2.CLIP을 여러 방식으로(kd, 모듈 등등)으로 추가학습하여 교정(대규모 데이터 필요)하는 방식
    3.Scratch부터 CLIP like 방식을 학습하는 방식 혹은 아예 다른 방식으로 학습하는 연구
    4.CLIP 대신 다른 모델로 대체하거나, 다른 모델을 같이 사용하거나, 일부를 가져와서 사용하는 방식
- 현재 많은 연구에서 사용되고 있는 CLIP은 언젠가 대체가 될 것인가? (4년이면 오래했나?), Retrieval과 같이 deterministic한 피처가 중요한 영역은 여전히 CLIP이 중요할 것으로 생각을 하였으나, 이 논문은 local 피처를 잘 파악하는 DINO를 사용하여, CLIP ViT보다 우수한 성능을 보였기 때문에, 지금과 같은 범용성은 낮아질 것으로 판단됨