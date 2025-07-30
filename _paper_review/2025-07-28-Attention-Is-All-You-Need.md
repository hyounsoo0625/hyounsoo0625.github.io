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

<img src="/images/paper_review/attention-is-all-you-need/bytenet.png" class="post_img"/>

<img src="/images/paper_review/attention-is-all-you-need/ConvS2S.png" class="post_img"/>

이 모델들은 CNN을 사용하며, input과 output의 모든 위치에 대해 hidden representation을 병렬로 계산한다. 이는 서로 떨어진 임의의 input 또는 output 위치 사이의 정보를 연결하기 위해 필요한 연산 횟수가 위치 간 거리의 증가에 따라 달라지며, ConvS2S의 경우 linear, ByteNet의 경우 logarithmic으로 증가하여 떨어진 위치 간 dependency를 학습하는 것이 더 어려워진다.

Transformer에서는 이러한 연산이 constant number의 연산으로 줄어들지만, attention-weighted된 위치들을 평균 내는 방식으로 인해 effective resolution이 감소하는 단점이 있다. 즉, 쉽게 풀어 얘기하면 self-attention은 RNN이나 CNN과 달리 멀리 떨어진 단어들 간의 관계를 연결하기 위해 필요한 연산의 수가 상수가 된다. 즉, 문장의 첫 단어와 마지막 단어를 연결하더라도 계산 단계는 늘어나지 않는다. self-attention은 '나'라는 단어가 '밥', '먹었다', '오늘' 등 여러 단어를 보고 각각의 중요도에 따라 평균적으로 정보를 가져온다. 이로 인해, 단여별로 고유하게 중요한 정보를 뽑기보다 평균적으로 섞인 정보를 얻게 된다. 그래서 중요한 세부 정보가 흐려지거나 낮아지는 일이 생길 수 있다.이는 [`Multi-Head Attention`](#multi-head-attention)을 사용해 보완하게 된다.

### Self-Attention
self-attention은 하나의 sequence 내에서 다른 위치 간의 관계를 파악해 그 sequence의 표현을 계산하는 attention mechanism이다.
Transformer는 sequence-aligned RNN이나 convolution을 전혀 사용하지 않고, self-attention만을 이용해 입력과 출력의 표현을 계산하는 transduction model이다.

# Model Architecture
대부분의 neural sequence transduction model은 encoder-decoder 구조로 구성되어 있다. 즉, input sequence로 \\((x_1, ..., x_n)\\)이 주어지면 연속된 표현인 \\(z=(z_1, ..., z_n)\\)으로 변환되고, \\(z\\)를 decoder에 주어지면 output sequence인 \\((y_1, ..., y_m)\\)을 한번에 하나씩 생성한다. 이때, 각 단계에서 model은 auto-regressive이며, 다음을 생성할 때, 이전에 생성된 symbol을 추가 입력으로 사용한다.
아래의 구조는 transformer의 전체 구조를 나타낸 그림이다. 이때, 왼쪽과 오른쪽에 각각 encoder와 decoder에 대해 stacked self-attention과 point wise, fully connected layer을 사용한다.

<img src="/images/paper_review/attention-is-all-you-need/transformer.png" class="post_img"/>

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

<img src="/images/paper_review/attention-is-all-you-need/attention.png" class="post_img"/>

attention의 입력은 차원이 \\(d_k\\)인 query와 key, 차원이 \\(d_v\\)인 value로 구성된다.
이때, query와 모든 key 간의 dot product을 계산한 후, 각 계산 결과를 \\(sqrt(d_k)\\)로 나눈다. 이후, softmax function을 적용해 각 value에 대한 weight을 얻는다.
실제로는 여러개의 query를 동시에 처리하기 위해 query, key, value을 하나의 행렬인 \\(Q\\), \\(K\\), \\(V\\)에 묶어서 attention 연산을 수행한다. 이때 출력 행렬은 다음과 같이 계산된다.

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

일반적으로 사용하는 attention 함수는 [`additive attention`](#additive-attention)과 [`dot-product(multiplicative) attention`](#dot-product(multiplicative)-attention)을 사용한다.

#### dot-product(multiplicative) attention
기존 사용하는 방식과 거의 동일하지만 차이점으로 scaling factor인 \\(\sqrt{d_k}\\)가 없다는 차이점이 있다.

#### Additive Attention
<img src="/images/paper_review/attention-is-all-you-need/additiveAttention.png">

additive attention은 1개의 hidden layer을 가진 feed-forward neural network를 사용해 유사도를 계산한다.

위 두 방식은 이론적으로 복잡도가 비슷하지만 (\\(O(n^2)\\)) dot-product attention은 최적화된 matrix multiplication으로 쉽게 구현할 수 있기 때문에 매우 빠르고 memory 효율도 높다.

이때, \\(d_k\\)가 작은 경우에는 두 방식의 성능이 유사하게 나오지만 \\(d_k\\)가 클수록 additive attention이 scaling이 없는 dot-product attention보다 더 좋은 성능을 낸다. 그 이유는 \\(d_k)\\)가 커질 수록 dot-product의 값이 매우 커져, softmax 함수가 gradient가 거의 0에 가까운 영역으로 진입한다고 추측한다. 이를 완화하기 위해 dot-product 값을 \\(\frac{1}{\sqrt{d_k}}\\)으로 나누어 scaling을 수행한다.

### Multi-Head Attention
이 논문에서는 query, key, value을 각각 \\(d_{model}\\)의 차원을 그대로 사용하는 대신, \\(h\\) (multi head 수)개의 서로 다른 학습 가능한 linear projection을 통해 각각의 query, key, value을 \\(d_k, d_k, d_v\\)의 차원으로 Linear projection을 하는 것이 더 효과적임을 확인했다. 이를 통해 attention을 병렬로 수행하면 각 head에서 \\(d_v\\)차원의 출력이 생성된다. 이를 concatenate한 후, linear projection을 적용하면 최종 attention의 결과가 된다. 이를 통해 서로 다른 위치에 있는 정보들을 동시에 주목(attend) 할 수 있게 해준다.

이전에 말했다시피 단일 attention head만 사용하면 weight averaging 때문에 표현력이 제한된다.

$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
\\
\quad where\; head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
이때, 사용되는 project 행렬은 다음과 같다.
- \\(W_i^Q \in R^{d_{model \times d_k}}\\)
    - query projection
- \\(W_i^K \in R^{d_{model \times d_k}}\\)
    - key projection
- \\(W_i^V \in R^{d_{model \times d_v}}\\)
    - value projection 
- \\(W^O \in R^{hd_v \times d_{model}}\\)
    - final projection

이 논문에서는 총 \\(h = 8\\)개의 병렬 attention layer을 사용한다. 이때, 각 head는 \\(d_{model}\\)이 아니라 다음과 같은 차원을 사용한다.

$$
d_k = d_v = \frac{d_{model}}{h} = 64
$$

즉, \\(d_model = 512\\)라고 가정하면 head는 64차원이 된다. 이를 통해 전체 multi-head attention의 게산 비용은 single head attention과 비슷한 수준으로 유지할 수 있다.

### Applications of Attention in our Model
> Transformer는 multi-head attention을 다음 세가지 방식으로 사용한다.
#### Encoder-Decoder Attention(Cross-Attention)
Encoder-Decoder Attention Layer에서 key와 value는 encoder의 출력에서 가져온다. 이렇게 함으로써, decoder의 모든 위치는 input sequence 전체 위치에 attention을 기울일 수 있다.
> 즉, encoder는 입력 전체를 요약하고, decoder는 그 요약 정보에 접근하여 출력 단어를 생성한다.
#### Encoder Self-Attention
encoder에는 self-attention layer가 포함되어 있다. 이 self-attention layer에서는 query, key, value가 모두 동일한 입력에서 나온다. 이 경우, encoder 내부에서 이전 layer의 출력이 다음 layer의 입력이 된다. 따라서 encoder 내의 각 위치는 encoder 이전 layer의 모든 위치에 주의를 기울일 수 있다.
> 즉, encoder는 입력 sequence 전체의 단어들 간 관계를 스스로 학습하며, 순서에 얽매이지 않고 모든 위치를 참고할 수 있다.
#### Decoder Self-Attention (with Masking)
마찬가지로, decoder 내의 self-attention layer는 decoder의 각 위치가 그 위치까지의 모든 decoder 위치에 주의를 기울일 수 있다. 하지만 auto-regressive 속성을 유지하기 위해 오른쪽(미래 방향)의 정보가 왼쪽(과거 위치)으로 흘러가는 것(leftward information flow)을 막아야 한다. 이를 위해 scaled dot-product attention 내부에서 masking을 수행한다. 즉, softmax 입력에서 허용되지 않은 연결(미래 정보)에 해당하는 값들은 \\(-\inf\\)로 설정하여 softmax의 결과가 0이 되도록 만든다
> 즉, decoder는 다음 단어를 예측할 때, 아직 생성되지 않은 미래 단어들을 참고하지 못하도록 제한한다.

## Position-wise Feed-Forward Networks
attention sub-layers 외에도, encoder와 decoder의 각 layer에는 fully connected feed-forward network를 포함하고 있으며, 이 네트워크는 입력 sequence의 각 위치에 개별적으로 그리고 동일한 방식으로 적용된다. 이 feed-forward network는 2개의 linear transformation으로 구성되어 있으며, 그 사이에 ReLU activate function이 적용된다. 수식으로는 다음과 같이 표현된다.

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$
- \\(x\\): input vector
- \\(W_1, W_2\\): weight matrix
- \\(b_1, b_2\\): bias vector
- \\(max(0, ...)\\): ReLU activate function

이 linear  transformation은 서로 다른 위치에 대해서는 동일한 방식으로 적용되지만, layer가 다르면 서로 다른 parameter을 사용한다. 이 구조는 kernel 크기 1짜리 convolution 연산 2개로도 표현할 수 있다.
쉽게 얘기하면 입력 sequence 각각의 token vector에 대해 동일한 계산을 반복한다는 것이 word1, word2, ...에 모두 동일한 FFN으로 계산되고, 각각 layer마다 다른 독립된 FFN을 사용한다.
> 즉, 각 위치마다 독립적으로 적용되는 연산이다.

입력과 출력의 차원은 \\(d_{model} = 512\\)이고, hidden layer의 차원은 \\(d_{ff} = 2048\\)이다. 즉, input vector는 먼저 2048차원으로 확장된 후, 다시 512차원으로 줄어든다.

## Embeddings and Softmax
다른 sequence transduction model들과 마찬가지로 학습된 embedding을 사용해 input token과 output token을 차원 \\(d_{model}\\)을 갖는 vector로 변환한다. 또한 일반적으로 사용되는 학습된 linear transformation과 softmax function을 사용하여 decoder의 출력을 다음 token의 확률 분포로 변환한다.
> 즉, decoder의 마지막 출력은 V차원으 확률 분포로 바뀌며, V는 vocabulary 크기가 된다.

transformer에서는 input embedding layer과 output embedding layer, softmax 이전의 linear transformation 사이에 동일한 weight matrix을 공유한다. 즉, 세군데에서 동일한 weight를 공유함으로써 parameter의 수를 줄이고 일반화 성능을 높이는 효과를 기대한다.
embedding layer에서는 해당 weight에 \\(\sqrt{d_{model}}\\)을 곱해준다. 이 scaling은 embedding 값의 분산을 조절하기 위한 것으로, 초기 학습 안정성과 성능 향상에 도움을 줄 수 있다.

## Positional Encoding

# Why Self-Attention

# Training

## Training Data and Batching

## Hardware and Schedule

## Optimizer

## Regularization

# Result

## Machine Translation

## Model Variations

## English Constituency Parsing

# Conclusion