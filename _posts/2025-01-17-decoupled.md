---
title: [Paper Review] Decoupled Knowledge Distillation
author: invhun
date: 2025-01-17 01:30:00 +0800
last_modified_at: 2025-01-17 01:30:00 +0800
categories: [Paper Review, Knowledge Distillation]
tags: [Knowledge Distillation]
pin: true
---


> Decoupled Knowledge Distillation   
> CVPR 2022   
> Borui Zhao, Quan Cu, Renjie Song, Yiyu Qiu, Jiajun Liang   
> MEGVII Technology, Waseda University, Tsinghua University   
> [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf)] [[github](https://github.com/megvii-research/mdistiller)]


# Abstract

최근 KD방식은 중간 레이어에서 깊은 피처 증류에 중점을 두며, 로직 증류의 중요성이 간과됨

저자는 기본 KD loss를 Target class knowledge distillation ***(TCKD)***, non-target knowledge distillation ***(NCKD)***로 재구성

***TCKD:*** 학습 샘플의 “난이도”와 관련된 지식을 전달

***NCKD:*** 로직증류가 작동하는 주요 이유

저자는 기본 KD loss가
***(1)NCKD의 효과를 억제함***

***(2)두 파트 간의 결합이 유연성을 제한함***

저자는 ***TCKD***와 ***NCKD***를 균형을 맞춘 방식 ***(Decoupled Knowledge Distillation) DKD***를 제안

CIFAR-100, ImageNet, MS-COCO에서 우수한 결과를 보임

# 1. Introduction

- 로직 증류는 피처 증류에 비해 적은 계산 비용을 요구하지만, 상대적으로 성능이 부족함
- 저자는 이러한 로직 증류의 잠재력이 제한되어 있다고 가정하여, KD의 매커니즘을 분석
- TCKD는 binary로 지식을 전이하며, 타겟 클래스의 예측만 제공이 됨
→ 따라서 학습 샘플의 “난이도”에 대한 지식을 전송한다고 가정할 수 있음
→ 이를 검증하기 위해, 난이도를 증가시키는 세 가지 측면의 연구를 수행: stronger augmentation, noiser label, inherently challenging dataset
- NCKD는 비타켓 로직 간의 지식 만을 고려하며, NCKD만 적용하여도 기존 KD보다 나은 결과를 얻을 수 있음을 실험적으로 증명하였음
→ NCKD로 증류되는 지식은 중요한 “dark knowledge”로 볼 수 있음

> *Dark knowledge: 주로 비타겟 클래스에 대한 예측 확률로, soft label를 통해 전이되며 단순한 hard 정답 레이블 이상의 내용을 담고있으며, 각 클래스 간의 상대적인 신뢰도와 불확실성을 포함함
→ 이로 인해 학생 모델이 단순히 정답 레이블만을 학습하는 것이 아니라, 클래스 간의 관계와 패턴을 이해하게 되어 일반화 능력 향상이 이루어짐
> 
> 
> ![캡처.PNG](https://1drv.ms/i/s!AvuRV8CuQlavgQLfA2CzRJS1mQ4J?embed=1&width=884&height=627)
> 
> 출처: [https://www.ttic.edu/dl/dark14.pdf](https://www.ttic.edu/dl/dark14.pdf)
> 

- 기존 KD는 이 두 파트의 결합된 방식으로 한계가 존재함
    - NCKD의 로스는 타겟 클래스에 대한 교사의 예측 신뢰도와 음의 상관계수로 가중치가 부여됨
    → 예측 점수 가 클수록, 가중치가 작아져서 NCKD의 효과를 억제함
    - TCKD와 NCKD를 별도의 가중치로 조정할수 없고 결합되어 있음
- DKD: 교사의 신뢰도와 NCKD 손실의 상관관계를 분리, NCKD와 TCKD도 분리하여, 중요도를 개별적으로 고려할 수 있도록 조정

## 기존 연구 문제점:

- 최신 지식 증류 방법은 중간 레이어의 깊은 특징에 집중하여 로직 증류의 중요성을 간과함
- 피처 기반 방법은 높은 계산 비용을 초래함
- 기존 KD 로스는 결합된 형식으로 NCKD의 효과를 억제하고, TCKD와 NCKD의 유연성을 제한함

## 제안 방법:

- TCKD와 NCKD로 KD 손실을 재구성하여, 각 부분의 효과를 분석하고 독립적으로 연구
- DKD를 제안하여 NCKD 로스를 교사의 신뢰도와 무관하게 설정하고, TCKD와 NCKD의 중요성을 개별적으로 고려함

# 2. Related work

- KD는 교사 네트워크가 더 작은 학생 네트워크의 훈련을 안내하는 학습 방식으로 정의됨
- “Dark knowledge”는 교사로부터 학새에세 소프트 레이블로 전이 되는 정보
- 온도 하이퍼파라미터: negative 로직에 대한, 중요도를 올리기 위해 사용됨
- 최근 방법들은 주로 중간 피처에 기반하고 있으며, 로직 기반 방법보다 높은 성능을 달성하지만, 상당히 높은 계산 비용을 수반함
- 이 논문은 로직 기반 방법의 잠재력을 제한하는 요소를 분석하고, 잠재력을 활성화 하는데 중점을 둠

# 3. Rethinking Knowledge Distillation

## 3.1. Reformulating KD

- t번째 Class에 속하는 샘플에 대해서, C개의 클래스가 있을 때, 각 $p_i$는 softmax함수로 계산된 i번째 클래스로 분류될 확률을 뜻함

![image.png](https://1drv.ms/i/s!AvuRV8CuQlavgQQjGZNXvsLtZvHR?embed=1&width=459&height=92)

- 정답 레이블인 target class와 관련성으로 분리하면, 아래와 같이 분리됨. 이때 $\bold{b} = [p_t, p_{\backslash t}]$, $p_t$는 타겟 클래스의 확률, $p_{\backslash t}$는 타겟 클래스를 제외한 확률의 합으로 softmax 특성 상 $p_{\backslash t}=1-p_t$가 성립됨
    
    ![fo1-1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavfr2Uyzy6VP6YXiw?embed=1&width=545&height=109)
    

- 또한 타겟 클래스를 제외하였을 때, i번째 클래스의 확률 $\hat{p_i}$는 아래와 같이 구함
    
    ![fo2.PNG](https://1drv.ms/i/s!AvuRV8CuQlavcgYQaPDLugzt1Q4?embed=1&width=468&height=91)
    
- 기존 KD의 경우 (3)번식으로 표현하며, 이를 $\mathcal{T}, \mathcal{S}$는 각각 Teacher와 Student를 의미함. KL divergence에서 타겟 클래스를 분리하여 아래와 같이 표현할 수 있음
    
    ![fo3.PNG](https://1drv.ms/i/s!AvuRV8CuQlavb2w4OzltYSJtqes?embed=1&width=561&height=159)
    
- 이때 (1)과 (2)의 식을 참고하면, $\hat{p_i}=p_i/p_{\backslash t}$임을 알 수 있음. 따라서 식 (3)의 $p_i$를 $\hat{p_i}$와 $p_{\backslash t}$의 곱으로 다시 표현하고, 정리하면 타겟 클래스에 대한 이진 확률에 관한 KL divergence식과 논 타겟 클래스 간 확률에 KL divergence 식으로 분리할 수 있음
    
    ![fo4.PNG](https://1drv.ms/i/s!AvuRV8CuQlaveIF-KOxUVsufT2g?embed=1&width=641&height=248)
    
- 저자는 전자를 Target Class Knowlege Distillation(TCKD), 후자를 Non-Target Class Knowledge Distillation(NCKD)로 명명하였고, 이를 표현하면
    
    ![fo5.PNG](https://1drv.ms/i/s!AvuRV8CuQlavd4khrcHv_9PYeV0?embed=1&width=584&height=59)
    
    ![fo6.PNG](https://1drv.ms/i/s!AvuRV8CuQlavfKmqQnXxQHvwN1s?embed=1&width=509&height=53)
    
- 즉 NCKD는 teacher의 타겟 클래스에 대한 확률이 높아질수록, NCKD의 가중치가 낮아짐을 확인할 수 있음. 또한 TCKD와 NCKD 각각을 유연성있게 균형을 맞추어 조정할 수가 없음

## 3.2. Effects of TCKD and NCKD

![t1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavbpvD6B1frzrzOGU?embed=1&width=649&height=638)

- 학생 모델과, 기존 KD, 그리고 TCKD, NCKD를 각각 사용하였을 때 성능 결과를 비교
- NCKD의 성능이 KD에 비해서 우세하며, TCKD의 성능은 베이스라인보다도 저조함
→ 타겟 클래스 관련 지식이 비타겟 클래스 간의 지식만큼 중요하지 않을 수 있음을 시사함

![t2.PNG](https://1drv.ms/i/s!AvuRV8CuQlavdOySkMuZa7VNtKE?embed=1&width=650&height=596)

![t4.PNG](https://1drv.ms/i/s!AvuRV8CuQlaveiD9je-sRkIQw7M?embed=1&width=556&height=140)

- TCKD는 단순 이진 분류 작업 지식을 전이하기 때문에, “difficulty”에 관한 지식을 전달한다고 볼 수 있음, CIFAR-100의 경우는 “difficulty”에 관한 지식이 유용하지 않을 수 있으므로, 세가지 관점의 테스트를 추가로 수행함
- (1)Applying Strong Augmention: AutoAugment를 사용하여 평가를 수행
→ (표2)강력한 증강을 사용할 경우, 성능 향상이 존재
- (2)Noisy Labels: 여러 비율로 noisy label을 추가
→(표3) 노이즈가 증가할수록, 성능 향상이 두드러짐
- (3)Challenging dataset(ImageNet): 더 어려운 데이터셋일 경우
→ CIFAR 100의 성능 하락과는 반대로 성능 향상을 기록

![t5.PNG](https://1drv.ms/i/s!AvuRV8CuQlavfa8mqLzuDnvpCq4?embed=1&width=645&height=192)

![fo6.PNG](https://1drv.ms/i/s!AvuRV8CuQlavfKmqQnXxQHvwN1s?embed=1&width=509&height=53)

- 식(6)을 통해 NCKD가 교사의 타겟 클래스에 대한 신뢰도(확률)이 높아질수록, 가치있는 “Dark knowledge”의 distill이 억압됨을 확인할 수 있음
- (표5)는 $p^T_t$를 기준으로 상위 50%, 하위 50%에 대해 각각 NCKD를 적용한 결과로, 상위 50% 샘플에 NCKD를 적용하였 을 때 더 높은 성능을 보임을 확인할 수 있음
→ 잘 예측된 샘플의 지식이 다른 샘플보다 풍부함을 시사하지만, 이러한 지식이 교사의 높은 신뢰도에 의해 억제됨

## 3.3. Decoupled Knowledge Distillation

- 직관적으로 TCKD와 NCKD는 모두 필수적이고 중요하지만, 기존 KD 공식에서는 TCKD와 NCKD가 결합되어 있음
(1) NCKD는 $1-p^T_t$와 결합되어, 잘 예측된 샘플에서 효과가 억제됨
(2) NCKD와 TCKD 각각의 가중치를 균형있게 조정할 수 없음. 두 항목의 기여가 서로 다른 측면에서 나오므로 별도로 고려해야함
- 각각의 가중치를 별도로 조정하는 방식 Decoupled Knowledge Distillation(DKD)를 제안함
    
    ![fo7.PNG](https://1drv.ms/i/s!AvuRV8CuQlave56eT2zM9IS7qPM?embed=1&width=433&height=44)
    
- 간단하지만 슈도 코드도 제공
    
    ![a1.PNG](https://1drv.ms/i/s!AvuRV8CuQlavc3NVf5HzfNMf0U4?embed=1&width=568&height=652)
    

# 4. Experiments

## 4.1. Main Results

### Ablation: $\alpha$ and $\beta$

![Screenshot 2025-01-16 at 11.48.07 PM.png](https://1drv.ms/i/s!AvuRV8CuQlavgQB7bzeXxsnE_7P-?embed=1&width=1112&height=262)

- NCKD의 가중치가 8.0, TCKD의 가중치가 1.0일 때 최고 성능을 달성, 기존 대비 2.59% 성능 향상
NCKD의 영향이 중요하지만, TCKD도 필수적

### CIFAR-100 image classification

![Screenshot 2025-01-16 at 11.50.34 PM.png](https://1drv.ms/i/s!AvuRV8CuQlavcalH-x_IZg3GfgU?embed=1&width=2350&height=778)

- DKD가 KD에 비해 일관된 성능 향상을 보임
- 또한 교사와 학생이 서로 다른 구조일 때, 더 큰 성능 향상을 기록

### ImageNet image classification

![Screenshot 2025-01-16 at 11.53.14 PM.png](https://1drv.ms/i/s!AvuRV8CuQlavgQGGHbHONDK1DOgY?embed=1&width=2338&height=668)

- SOTA급 성능을 보이며, CIFAR 100과 비슷한 양상의 결과를 보임

### MS-COCO object detection

![Screenshot 2025-01-16 at 11.55.33 PM.png](https://1drv.ms/i/s!AvuRV8CuQlaveVMkMVz6rdVdnW4?embed=1&width=2344&height=710)

- DKD가 기존 KD의 성능을 상회하나, object detection은 깊은 feature의 품질에 크게 의존하기 때문에, DKD단독으론 높은 성능을 달성할 수 없음
- 따라서 feature기반 증류 방식인 ReviewKD와 결합하여 SOTA 성능을 달성

## 4.2. Extension

### Training efficiency

![Screenshot 2025-01-16 at 11.59.14 PM.png](https://1drv.ms/i/s!AvuRV8CuQlavgQN3RPgNSWTw9erk?embed=1&width=1134&height=830)

- DKD는 기존 KD를 재구성한 것이기 때문에, 거의 동일한 계산 복잡도를 필요로 하며, 추가 파라미터를 요구하지 않음 ↔ 피처 기반 증류 방식은 추가 훈련시간과 GPU메모리 비용이 발생

### Improving performances of big teachers

![Screenshot 2025-01-17 at 12.04.00 AM.png](https://1drv.ms/i/s!AvuRV8CuQlavdqLWo7pTT8sZrQ0?embed=1&width=1168&height=682)

- 더 큰 모델이 항상 더 나은 교사가 아니라는 현상이 존재, 이전 연구는 큰 교사와 작은 학생 간의 큰 용량 차이로 이 현상을 설명하였음
- 저자는 기존 KD 로스식의 NCKD의 억제, 즉 ($1-p^T_t$)가 교사가 커질수록 작아지기 때문이라고 주장
- 이는 기존 연구의 실험결과와도 일맥상통하며 표 11과 12의 결과에서도 일관되게 나타남

### Feature transferability

![Screenshot 2025-01-17 at 12.05.22 AM.png](https://1drv.ms/i/s!AvuRV8CuQlavcDCYsyPfg6Ft5rU?embed=1&width=1124&height=274)

- CIFAR-100 데이터셋에서 KD를 수행 후, feature 전이를 평가하는데 유용한 downstream task인 linear probing task를 수행
- 그 결과 DKD가 더 일반화 가능한 지식을 전이하는 것을 검증

### Visualizations

![Screenshot 2025-01-17 at 12.09.25 AM.png](https://1drv.ms/i/s!AvuRV8CuQlavf2HF2FgVJWndMR0?embed=1&width=1138&height=996)

- 그림3: TSNE결과 DKD의 representation이 KD보다 더 분리 가능함을 보여줌
- 그림4: 학생과 교사 로직의 상관 행렬 차이를 시각화 한 것으로, DKD는 학생이 교사와 더 유사한 로직을 출력하도록 도와줌

# 5. Discussion and Conclusion

- 기존 KD로스를 TCKD, NCKD 두 부분으로 재구성하고, 각각의 효과를 분석하고 입증함
- 결합된 공식이 효율과 유연성을 제한함을 발견하였고, 이를 해결하고자 분리된 방식인 DKD를 제안

### limitation

- object detection에서는 피처 기반 방법인 reviewKD보다 낮은 성능을 기록
- 증류 성능과 $\beta$와의 상관관계는 완전히 조사되지 않았음