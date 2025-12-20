---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
layout: single
collection: paper_review
author_profile: true
---

# Abstract
이 논문은 제한된 input view만 사용해 복잡한 scene을 합성하는데 sota을 내는 방법론은 제안하였다. 이는 장면을 하나의 연속적은 continuous volumetric scene function으로 표현하고, 이는 convolution을 사용하는 것이 아닌 단순한 fully-connected deep network을 통해 구현된다. 이는 하나의 연속적은 5D coordinate (공간위치인 $x, y, z$와 viewing direction $\theta, \phi$\)을 입력받아 해당 공간 위치에서 volume density와 view-dependent emitted radiance를 출력한다.

- volume density
    - 여기에 무언가 입자가 있나? $\rightarrow$ volume density($\sigma$)
    - 특정 3D 좌표 $\(x, y, z\)$에 얼마나 불투명한 물질이 있는지 나타내는 scalar 값
    - $\sigma$가 높은면 그 지점은 불투명함 \rightarrow 빛이 이 지점을 통과하기 어렵고, 여기에 부딪혀 흠수되거나 반사될 확률이 높음
    - $\sigma$가 낮으면 그 지점은 투명함
- view-dependent emitted radiance
    - 만약 있다면 특정 방향에서 볼 때 무슨 색인가? $\rightarrow$ view-dependent emitted radiance $c$
    - 3D 좌표 $\(x, y, z\)$에서 특정 방향으로 방출되는 빛의 색상(RGB) 값
    - 3D 위치 뿐만 아니라 보는 뱡향에도 의존

새로운 view을 합성할 때는 camera rays을 따라 여러 5D coordinate을 query와 출력된 색과 density을 volume rendering 기법으로 이미지를 projection한다. 이 논문에서는 neural radiance fields을 효과적으로 optimize해 복잡한 기하와 외관을 가진 장면들의 새로운 view을 rendering 할 수 있는지 설명하고, 기존의 neural rendering과 view synthesis task보다 우수한 결과를 보이는 것을 보인다.

# Introduction
해당 연구에서는 continuous 5D scene representation의 parameter을 직접 최적화혀여 촬영된 이미지 set을 rendering 하였을 때 오차를 최소화 하는 방식으로 문제를 해결했다고 한다. 정적으로 된 scene을 하나의 연속적인 5D function으로 표현하는데 이는 공간의 각 지점 $\(x, y, z\)$에서 각 방향 $\(\theta, \phi\)$로 방출되는 radiance와 그 지점의 density을 출력한다. 이 density는 differential opacity controlling처럼 동작하여 한 ray가 $\(x, y, z\)$를 통과할 때 얼마만큼의 radiance가 누적되는지를 제어한다.

> **differential opacity controlling** <br/>
NeRF는 ray가 3D space을 통과할 때, 그 경로를 따라 연속적인 적분을 수행한다. 실제로는 이 ray을 잘게 쪼개어 여러 개의 작은 segment로 나누고, 각 구간의 값을 합산한다. 여기서 $\sigma$는 이 광선이 아주 작은 구간을 지날 때, 이 구간에서의 불투명도가 된다. 만약 $\sigma$가 높으면(불투명하면) 해당 지점의 색상$\(c\)$가 최종 pixel 색상에 더 많은 비중을 차지하게 된다.

해당 연구는 MLP을 사용해 5D coordinate $\(x, y, z, \theta, \phi\)$로부터 하나의 volume density와 view-dependent RGB color를 예측하도록 한다.
특정 viewpoint에서 NeRF를 rendering하려면 3개의 step이 필요하다.

<img src="images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig2.png" />

1. camera ray들을 scene에 따라가며 sampling된 3D 점들의 집합을 생성하고, 2. 그 점들과 해당하는 2D viewing directions을 신경망의 입력으로 넣어 색과 밀도 출력을 얻은 뒤, 3. 고전적인 volume rendering 기법으로 이 색과 밀도들을 누적하여 2D 이미지를 만든다. 이 과정은 미분 가능하기 때문에, 관측된 각 image와 해당 표현으로부터 rendering한 view간 오차를 경사하강법으로 최소화하여 모델을 최적화 한다.

# Related Work
## Neural 3D shape representations
x, y, z 좌표를 signed distance function이나 occupancy field로 mapping하는 deep network을 최적화해 연속적인 3D shape를 level set으로 암묵적으로 표현하는 방법을 탐구하였다. 하지만 이는 ShapeNet과 같은 합성 3D shape dataset에서 얻은 gt 3D geometry에 접근할 것을 요구하는 한계가 있다. 이후 연구들은 `differentiable rendering function`을 정식화해 2D image만으로도 최적화 할 수 있게 하여 gt 3D shape 요구를 완화했다. `Niemeyer`은 표면을 3D occupancy field로 표현하고, 각 ray에 대해 표면과의 교차점을 찾은 뒤, implicit differentiation을 통해 derivative(도함수)을 계산한다. 해당 광선의 교차 위치는 neural 3D texture field의 입력으로 주어져 그 지점의 diffuse color을 예측한다.
`Sitzmann`은 각 연속된 3D coordinate에서 feature vector와 RGB color를 출력하는 neural 3D 표현을 사용하고, 각 ray에 따라 어디에 표면이 있는지 결정하는 RNN 기반 differentiable rendering fuction을 제안했다. 이는 이론적으로 복잡하고 고해상도 기하를 표현할 수 있는 잠재력을 지녔지만 기하학적 복잡성이 낮은 단순한 형태에만 제한되어 과도하게 oversmoothed rendering을 초래하는 경우가 많았다.

## View synthesis and image-based rendering
dense한 view sampling이 주어지면 단순한 light field sample interpolation 기법으로도 photorealistiv한 novel view을 재구성할 수 있다. 하지만 view sample이 드물 경우 전통적인 geometry와 appearance 표현을 관측 이미지로부터 예측하는 방향으로 큰 진전이 있었다. 또 다른 방법으로 volumetric representation을 사용해 RGB image들로부터 고품질의 photorealistic novel view을 생성하는 것이다. Volumetric 접근법은 복잡한 형태와 재질을 현실적으로 표현할 수 있고, gradient 기반 최적화에 적합하며 mesh기반 방법보다 시각적 noise를 덜 일으키는 경향이 있다.
> **Volumetric과 mesh기반 방법론**<br/>
Mesh는 3D object의 표면을 표현하여 3개 이상의 정점으로 이루어진 평면을 사용하는 반면 Volumetric은 object가 차지하는 3D space 전체를 잘게 쪼개어 표현하는 방식으로 3D 공간상의 가장 작은 정육면체 단위

초기의 volumetric 방법들은 관측 이미지를 사용해 voxel grid에 직접 색을 입히는 방식을 사용하였고, 최근에는 다수의 scene dataset을 이용해 입력 image들로부터 sampling된 volumetric 표현을 예측하도록 deep network를 학습한 뒤, alpha-compositing이나 학습된 compositing을 통해 ray를 따라 novel view를 rendering하는 방법들이 나왔다. 또 다른 연구로 CNN과 sampling된 voxel grid를 특정 장면별로 함께 최적화혀여 CNN이 low-resolution voxel grid로 인한 discretization artifact를 보정하게 하거나, 시간 또는 애니메이션 제어에 따라 예측된 voxel grid가 변화하도록 허용하였다.

이들은 이산적 sampling 때문에 높은 해상도로 확장하는데 시간, 공간 복잡도 측면에서 근본적인 한계가 있다. 즉, 고해상도의 이미지를 rendering하려면 3D 공간을 더 세밀하게 sampling하여야 한다. 이를 해당 연구에서는 deep fully-connected neural network의 parameter 내에 연속적인 volume을 encoding하는 방식으로 구성한다.

# Neural Radiance Field Scene Representation
연속적인 장면을 3D location $x = \(x, y, z\)$와 2D viewing direction$\(\theta, \phi\)$을 입력으로 받고, 출력으로 emitted color인 $c = \(r, g, b\)$과 volume density $\sigma$를 내보내는 5D vector-valued function으로 표현한다. 실제로 방향을 3차원 unit vector인 $d$로 표현한다. 연속적인 5D scene 표현을 근사하기 위해 MLP network $F_{\Theta} : \(x, d\) \rightarrow \(c, \sigma\)$를 사용하고, 각 input 5D coordinate가 대응하는 volume density와 directional emitted color을 출력하도록 network의 weight인 $\Theta$을 최적화한다.
장면 표현이 다중 시점에서 일관성을 유지하기 위해 다음을 만족시킨다.

- $\sigma$는 오직 위치 $x$에만 의존하도록 한다.
- RGB color $c$는 위치와 시점(방향) 둘 다의 함수로 예측할 수 있게 허용한다.

이를 위해 MLP $F_{\Theta}$는 먼저 입력된 3D 좌표 $x$를 8개의 fully-connected layer로 처리하고 각 layer는 ReLU activate function을 사용하고  256 dimension을 가진다. 이를 통해 $\sigma$와 길이 256의 feature vector을 출력한다. 이 feature vector는 camera ray의 시선 방향과 concatenate 된 뒤, 하나의 추가적인 fully-connected를 거쳐 RGB color 값을 출력한다.

x $\rightarrow$ fc $\times$ 8 $\rightarrow$ $\sigma$, feature vector

<img src="images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig3.png" />

위 그림을 보면 입력된 view direction을 어떻게 사용하여 non-Lambertian effects를 표현하는지 예시를 보여주고 있다. non-Lambertian effects는 보는 각도에 따라 물체의 색상이나 밝기가 달라지는 현상이다. lambertian 표면은 빛을 모든 방향으로 균일하게 분산시킨다. 따라서 어느 각도에서 보든 해당 지점의 색상과 밝기가 동일하기 보이는 반면 Non-Lambertian은 빛을 특정 방향으로 강하게 반사하여 보는 각도에 따라 표면의 색상이나 밝기가 크게 달라진다.

<img src="images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig4.png" />

위 그림은 view dependence 없이 즉, 입력으로 $x$만 사용했을 때 학습된 모델은 specularities를 잘 표현하지 못하는 어려움이 있다.

# Volume Rendering with Radiance Fields
해당 5D Neural Radiance Field는 scene의 임의의 공간 point에서의 volume density와 directional emitted radiance로 장면을 표현한다. 이는 classic한 volume rendering의 원리를 사용해 scene을 통과하는 임의의 ray의 색을 rendering한다. volume denstity $\sigma\(x\)$는 그 위치 $x$에 있는 미세한 입자에서 ray가 종료될  differential probability로 해석할 수 있다. camera ray $r(t) = o + td$가 근거리 경계 $t_n$에서 원거리 경계 $t_f$까지 지날 때의 기대 color $C\(r\)$는 다음과 같다.

$$
C(r) = \integral^{t_f}_{t_n} T(t)\sigma(r(t))c(r(t), d) dt, where T(t) = exp(-\integral^t_{t_n} \sigma(r(s))ds)
$$

여기서 $T(t)$는 t_n에서 t까지 ray가 다른 입자에 충돌하지 않고 통과할 누적 trasmittance, 즉 그 구간을 아무 입자와도 만나지 않을 확률을 나타낸다. 이 continuous integral을 수치적으로 추정하기 위해 quadrature을 사용한다. 이산화된 voxel grid rendering에 보통 쓰이는 quadrature는 MLP가 고정된 이산 위치 집합에서만 query되도록 하여 표현의 해상도를 사실상 제한하게 된다. 대신 해당 연구에서는 stratified sampling 방식을 사용한다. 이는 구간 $\[t_n, t_f\]$을 $N$개의 동일 간격 bin으로 분할한 뒤, 각 bin 안에서 하나의 sample을 random하게 뽑는다.

$$
t_i ~ U[t_n + \frac{i-1}{N} (t_f - t_n), t_n + \frac{i}{N} (t_f - t_n)]
$$

이는 연속 장면 표현을 유지할 수 있게 한다. 이 sample들을 사용해 Max의 volume rendering에서 논의된 quadrature 규칙으로 $C(r)$을 추정한다.

$$
\hat{C}(r) = \sum^N_{i=1} T_i(1-exp(-\sigma_i\lambda_i))c_i, where T_i = exp(-\sum^{i-1}_{j=1} \sigma_j \lambda_j)
$$

여기서 $\sigma_i = t_{i+1}-t_i$는 인접 sample 사이의 거리이다. 이 $\hat{C}(r)$를 $\(c_i, \sigma_i\)$ 값들로 계산하는 함수는 미분이 가능하고 $\alpha_i = 1 - exp(-\sigma_i \labmda_i$을 가지는 전통적인 alpha compositing으로 환원된다.

# Optimizing a Neural Radiance Field
이전 절에서는 기본 구성만으로는 sota의 성능을 내기에는 충분하지 않다. 해당 절에서는 고해상도의 복잡한 장면을 표현할 수 있도록 돕는 2가지 개선점을 도입했다.
1. input coordinate에 대한 positional encoding으로 MLP가 고주파의 성분을 더 잘 표현하도록 돕는다.
2. hierarchical sampling으로 고주파 표현을 효율적으로 sampling 할 수 있도록 돕는다.

## Positional encoding
이론적으로 neural network는 $x, y, z, \theta, \phi$를 입력으로 받아 동작하면 색상과 기하 구조의 고주파 변화를 잘 표현하지 못하는 출력이 나온다. 이는 deep network가 저주파 함수 학습에 편향된 경향이 있음을 보인다. 또한 입력을 고주파 함수들로 mapping하여 더 높은 차원의 공간으로 보낸 뒤 network에 전달하면 고주파 변화를 포함한 데이터를 더 잘 적합할 수 있다는 것을 보여줬다. 이를 $F_{\Theta} = F'_{\Theta} \circ \gamma$로 재정의하여 학습가능한 함수($F'_{\Theta}$)와 학습이 불가능한 함수($\gamma$)를 통해 성능이 크게 개선됨을 보였다. 여기서 $\gamma$는 $\mathbb{R}$에서 더 높은 차원의 공간인 $\mathbb{R}^{2L}$로의 mapping이며 encoding은 다음과 같이 구성되어 있다.

$$
\gamma(p) = (sin(2^0\pi p), cos(2^0\pi p), ..., sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))
$$

여기서 $\gamma{\cdot}$은 x의 세 좌표의 값이 각각 적용된다. 기존 transformer는 위치 정보를 제공하기 위해 사용한 것과 달리 NeRF에서는 입력 좌표를 더 높은 차원으로 mapping하여 MLP가 고주파 함수를 더 쉽게 근사하도록 만든다.

## Hierarchical volume sampling
각 camera ray에 대해 Neural Radiance Field를 N개의 query 지점에서 평가하는 방식은 비효율적이다. 이는 가려진 영역까지 반복해서 샘플링 하기 때문이다. 이는 최종 rendering에 미칠 것으로 기대되는 영향에 비례하여 sample을 할당하는 계층적 표현을 제안하였다. 이를 통해 단일 network을 사용하는 대신 2개의 network로 하나는 "coarse", 다른 하나는 "fine"을 최적화 한다. stratified sampling으로 $N_c$개의 위치를 sampling하고, 이들 위치에서 coarse network을 평가해 $\hat{C}(r)$을 얻는다. coarse network의 출력에 따라 ray에서 정보에 기반한 sampling을 생성하는데 이는 volume과 관련한 구간으로 편향되도록 한다. 이를 통해 다음과 같이 다시 쓴다.

$$
\hat{C}_c(r) = \sum^{N_c}_{i=1} w_i c_i, w_i = T_i(1-exp(-\sigma_i \lambda_i)).
$$

이를 정규화 하여 $\hat{w}_i = w_i / \sum_j w_j$로 만들면 ray상 구간별 확률 밀도 함수(PDF)가 나온다. 이를 통해 inverse transform sampling을 이용해 두 번째 집합의 $N_f$ 위치를 sampling하고, 처음과 두번째 sample의 합집합에서 fine network을 평가한다. 기르고 $N_c + N_F$개의 sample을 모두 사용해 최종 rendering color인 $\hat{C}_f(r)$을 계산한다. 이를 통해 가시적 content를 포함할 것으로 기대하는 영역에 더 많은 sample을 할당하므로 효율을 높일 수 있다.

## Implementation details
각 scene마다 별도의 neural continuous volume representation network을 최적화한다. 이를 위해 RGB image들, camera pose, intrinsic parameters, scene bounds가 필요하다. 각 최적화 iteration에서 dataset으리 모든 픽셀로부터 random으로 camera ray을 sampling한 뒤, hierarchical volume sampling을 따라 coarse network에서 $N_c$ sample을, fine network에서 $N_c+N_f$ sample을 query 한다. 이후, volume rendering 절차로 두 집합의 sample에서 각 ray의 색을 rendering 한다. Loss는 다음과 같다.

$$
L = \sum_{r \in R}[||\hat{C}_c(r) - C(r)||^2_2 + ||\hat{C}_f(r) - C(r)||^2_2]
$$
여기서 R은 각 batch의 ray의 집합이고, $C(r), \hat{C}_c(r), \hat{C}_f(r)$은 각각 실제, coarse 예측, fine 예측 RGB 색상이다. 

# Results
## Datasets
### Synthetic renderings of objects
dataset으로 2개의 synthetic renderings of object("Diffuse Synthetic 360'", "Realistic Synthetic 360'")에 대해 실험을 수행하였다. 