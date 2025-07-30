---
title: "Attention Is All You Need"
layout: single
collection: paper_review
author_profile: true
---

# Introduction
Recurrent neural networks(RNN), Long Short-term Memory(LSTM), gated recurrent neural networks는 language modeling, 기계 번역과 같은 sequence modeling 작업에서 sota를 달성하였다.
### Recurrent Models
Recurrent model은 input과 output의 symbol 위치에 따라 계산을 수행한다. 계산 시점에 위치를 정렬해 이전 hidden state인 \\(h_{t-1}\\)과 위치 \\(t\\)에 대한 입력의 함수로 hidden state인 \\(h_t\\)의 sequence를 생성한다. 이런 순차적인 특징으로 인해 훈련시 병렬화가 불가능하며, 메모리 제약 조건으로 인해 batching이 제한되므로 더 긴 sequence 길이에서 중요해진다.

최근 연구에서는 `factorization tricks`, `conditional computation`을 통해 계산 효율성을 크게 향상시켰으며, conditional computation은 모델의 성능도 함께 향상되었지만 순차적인 계산의 근본적인 제약은 여전히 남아있었다.

### Attention Mechanism
Attention mechanism은 다양한 task에서 설득력 있는 sequence modeling 과 transduction model의 필수적인 부분이 되었고, input과 output sequence에서 거리와 관계없이 dependency을 모델링 할 수 있게 되었지만 몇가지의 경우(*보기)를 제외하고 attention mechanism은 recurrent network와 함께 사용되었다.

이 논문에서는 Recurrence를 피하고 input과 output 간의 global dependency을 도출하기 위해 attention mechanism에 의존하는 model architecture인 `transformer`을 제안하였다. 이는 기존 RNN에서의 문제점인 병렬화를 허용해 8개의 P100 GPU에서 12시간이라는 짧은 시간동안 훈련된 translation 품질에서 sota를 도달하였다.

# Background
순차적 계산을 줄이려는 목표는 `Extended Neural GPU`, `ByteNet`, `ConvS2S` 모델들의 기반이 된다.

<img src="/images/paper_review/attention-is-all-you-need/bytenet.png" style="width: 50%;"/>

<img src="/images/paper_review/attention-is-all-you-need/ConvS2S.png" style="width: 50%;"/>

이 모델들은 CNN을 사용하며, input과 output의 모든 위치에 대해 hidden representation을 병렬로 계산한다. 이는 서로 떨어진 임의의 input 또는 output 위치 사이의 정보를 연결하기 위해 필요한 연산 횟수가 위치 간 거리의 증가에 따라 달라지며, ConvS2S의 경우 linear, ByteNet의 경우 logarithmic으로 증가하여 떨어진 위치 간 dependency를 학습하는 것이 더 어려워진다.

Transformer에서는 이러한 연산이 constant number의 연산으로 줄어들지만, attention-weighted된 위치들을 평균 내는 방식으로 인해 effective resolution이 감소하는 단점이 있다. 즉, 쉽게 풀어 얘기하면 self-attention은 RNN이나 CNN과 달리 멀리 떨어진 단어들 간의 관계를 연결하기 위해 필요한 연산의 수가 상수가 된다. 즉, 문장의 첫 단어와 마지막 단어를 연결하더라도 계산 단계는 늘어나지 않는다. self-attention은 '나'라는 단어가 '밥', '먹었다', '오늘' 등 여러 단어를 보고 각각의 중요도에 따라 평균적으로 정보를 가져온다. 이로 인해, 단여별로 고유하게 중요한 정보를 뽑기보다 평균적으로 섞인 정보를 얻게 된다. 그래서 중요한 세부 정보가 흐려지거나 낮아지는 일이 생길 수 있다.이는 [`Multi-Head Attention`](#multi-head-attention)을 사용해 보완하게 된다.

### Self-Attention
self-attention은 하나의 sequence 내에서 다른 위치 간의 관계를 파악해 그 sequence의 표현을 계산하는 attention mechanism이다.
Transformer는 sequence-aligned RNN이나 convolution을 전혀 사용하지 않고, self-attention만을 이용해 입력과 출력의 표현을 계산하는 transduction model이다.

# Model Architecture
대부분의 neural sequence transduction model은 encoder-decoder 구조로 구성되어 있다. 즉, input sequence로 \\((x_1, ..., x_n)\\)이 주어지면 연속된 표현인 \\(z=(z_1, ..., z_n)\\)으로 변환되고, \\(z\\)를 decoder에 주어지면 output sequence인 \\((y_1, ..., y_m)\\)을 한번에 하나씩 생성한다. 이때, 각 단계에서 model은 auto-regressive이며, 다음을 생성할 때, 이전에 생성된 symbol을 추가 입력으로 사용한다.
아래의 구조는 transformer의 전체 구조를 나타낸 그림이다. 이때, 왼쪽과 오른쪽에 각각 encoder와 decoder에 대해 stacked self-attention과 point wise, fully connected layer을 사용한다.

<img src="/images/paper_review/attention-is-all-you-need/transformer.png" style="width: 20%;"/>

## Encoder and Decoder Stacks
### Encoder
Encoder는 \\(N=6\\)개의 동일한 layer의 stack으로 구성되고, 각 layer에는 2개의 sub-layer가 있다. 1번째 sub-layer는 multi-head self-attention mechanism을 사용하고, 2번째 sub-layer는 간단한 positionwise fully connected feed-forward network을 사용한다. 각 sub-layers는 residual connection을 사용하고, 이후, normalization을 적용한다.

$$
LayerNorm(x + Sublayer(x))
$$

residual connection을 쉽게 하기 위해 sub-layer와 embedding layer는 \\(d_{model} = 512\\)인 출력을 내놓는다.

### Decoder
Decoder도 Encoder와 마찬가지로 \\(N = 6\\)개의 동일한 layer의 stack으로 구성된다. decoder는 encoder stack에 대해 multi-head attention을 수행하는 세번째 sub-layer을 삽입한다. 또한 decoder에서는 self-attetnion을 수정해 현재 이후 위치로 attention을 하는 것을 방지한다. 이러한 masking을 출력 embedding이 한 위치만큼 offset 된다는 사실과 결합되어 위치 \\(i\\)에 대한 예측이 \\(i\\)보다 작은 위치의 알려진 출력에만 의존할 수 있도록 한다.

## Attention
Attention은 query와 key-value 쌍의 집합을 출력에 mapping하는 것으로 설명할 수 있으며, query, key, value, output은 모두 vector의 형태로 구성되어 있다. Output은 value의 weight sum으로 계산되고 각 value에 할당된 weight는 해당 key을 사용한 query의 compatibility function을 통해 계산된다.

### Scaled Dot-Product Attention

<img src="/images/paper_review/attention-is-all-you-need/attention.png" style="width: 50%;"/>

attention의 입력은 차원이 \\(d_k\\)인 query와 key, 차원이 \\(d_v\\)인 value로 구성된다.
이때, query와 모든 key 간의 dot product을 계산한 후, 각 계산 결과를 \\(sqrt(d_k)\\)로 나눈다. 이후, softmax function을 적용해 각 value에 대한 weight을 얻는다.
실제로는 여러개의 query를 동시에 처리하기 위해 query, key, value을 하나의 행렬인 \\(Q\\), \\(K\\), \\(V\\)에 묶어서 attention 연산을 수행한다. 이때 출력 행렬은 다음과 같이 계산된다.

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

## Multi-Head Attention