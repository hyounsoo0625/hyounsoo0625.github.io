---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
layout: single
collection: paper_review
author_profile: true
---

# Abstract
이 논문은 제한된 input view만 사용해 복잡한 scene을 합성하는데 sota을 내는 방법론은 제안하였다. 이는 장면을 하나의 연속적은 continuous volumetric scene function으로 표현하고, 이는 convolution을 사용하는 것이 아닌 단순한 fully-connected deep network을 통해 구현된다. 이는 하나의 연속적은 5D coordinate (공간위치인 \\(x, y, z\\)와 viewing direction \\(\theta, \phi\\)\)을 입력받아 해당 공간 위치에서 volume density와 view-dependent emitted radiance를 출력한다.

- volume density
    - 여기에 무언가 입자가 있나? \\(\rightarrow\\) volume density(\\(\sigma\\))
    - 특정 3D 좌표 \\(\(x, y, z\)\\)에 얼마나 불투명한 물질이 있는지 나타내는 scalar 값
    - \\(\sigma\\)가 높은면 그 지점은 불투명함 \rightarrow 빛이 이 지점을 통과하기 어렵고, 여기에 부딪혀 흠수되거나 반사될 확률이 높음
    - \\(\sigma\\)가 낮으면 그 지점은 투명함
- view-dependent emitted radiance
    - 만약 있다면 특정 방향에서 볼 때 무슨 색인가? \\(\rightarrow\\) view-dependent emitted radiance \\(c\\)
    - 3D 좌표 \\(\(x, y, z\)\\)에서 특정 방향으로 방출되는 빛의 색상(RGB) 값
    - 3D 위치 뿐만 아니라 보는 뱡향에도 의존

새로운 view을 합성할 때는 camera rays을 따라 여러 5D coordinate을 query와 출력된 색과 density을 volume rendering 기법으로 이미지를 projection한다. 이 논문에서는 neural radiance fields을 효과적으로 optimize해 복잡한 기하와 외관을 가진 장면들의 새로운 view을 rendering 할 수 있는지 설명하고, 기존의 neural rendering과 view synthesis task보다 우수한 결과를 보이는 것을 보인다.

# Introduction
해당 연구에서는 continuous 5D scene representation의 parameter을 직접 최적화혀여 촬영된 이미지 set을 rendering 하였을 때 오차를 최소화 하는 방식으로 문제를 해결했다고 한다. 정적으로 된 scene을 하나의 연속적인 5D function으로 표현하는데 이는 공간의 각 지점 \\(\(x, y, z\)\\)에서 각 방향 \\(\(\theta, \phi\)\\)로 방출되는 radiance와 그 지점의 density을 출력한다. 이 density는 differential opacity controlling처럼 동작하여 한 ray가 \\(\(x, y, z\)\\)를 통과할 때 얼마만큼의 radiance가 누적되는지를 제어한다.

> **differential opacity controlling** <br/>
NeRF는 ray가 3D space을 통과할 때, 그 경로를 따라 연속적인 적분을 수행한다. 실제로는 이 ray을 잘게 쪼개어 여러 개의 작은 segment로 나누고, 각 구간의 값을 합산한다. 여기서 \sigma는 이 광선이 아주 작은 구간을 지날 때, 이 구간에서의 불투명도가 된다. 만약 \sigma가 높으면(불투명하면) 해당 지점의 색상\\(\(c\)\\)가 최종 pixel 색상에 더 많은 비중을 차지하게 된다.

해당 연구는 MLP을 사용해 5D coordinate \\(\(x, y, z, \theta, \phi\)\\)로부터 하나의 volume density와 view-dependent RGB color를 예측하도록 한다.
특정 viewpoint에서 NeRF를 rendering하려면 3개의 step이 필요하다.

<img src="/images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig2.png" class="post_img"/>

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
연속적인 장면을 3D location \\(x = \(x, y, z\)\\)와 2D viewing direction\\(\(\theta, \phi\)\\)을 입력으로 받고, 출력으로 emitted color인 \\(c = \(r, g, b\)\\)과 volume density \\(\sigma\\)를 내보내는 5D vector-valued function으로 표현한다. 실제로 방향을 3차원 unit vector인 \\(d\\)로 표현한다. 연속적인 5D scene 표현을 근사하기 위해 MLP network \\(F_{\Theta} : \(x, d\) \rightarrow \(c, \sigma\)\\)를 사용하고, 각 input 5D coordinate가 대응하는 volume density와 directional emitted color을 출력하도록 network의 weight인 \\(\Theta\\)을 최적화한다.
장면 표현이 다중 시점에서 일관성을 유지하기 위해 다음을 만족시킨다.

- \\(\sigma\\)는 오직 위치 \\(x\\)에만 의존하도록 한다.
- RGB color \\(c\\)는 위치와 시점(방향) 둘 다의 함수로 예측할 수 있게 허용한다.

이를 위해 MLP \\(\F_{\Theta}\\)는 먼저 입력된 3D 좌표 \\(x\\)를 8개의 fully-connected layer로 처리하고 각 layer는 ReLU activate function을 사용하고  256 dimension을 가진다. 이를 통해 \\(\sigma\\)와 길이 256의 feature vector을 출력한다. 이 feature vector는 camera ray의 시선 방향과 concatenate 된 뒤, 하나의 추가적인 fully-connected를 거쳐 RGB color 값을 출력한다.

x \\(\rightarrow\\) fc \\(\times\\) 8 \\(\rightarrow\\) \\(\sigma\\), feature vector

<img src="/images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig3.png" class="post_img"/>

위 그림을 보면 입력된 view direction을 어떻게 사용하여 non-Lambertian effects를 표현하는지 예시를 보여주고 있다. non-Lambertian effects는 보는 각도에 따라 물체의 색상이나 밝기가 달라지는 현상이다. lambertian 표면은 빛을 모든 방향으로 균일하게 분산시킨다. 따라서 어느 각도에서 보든 해당 지점의 색상과 밝기가 동일하기 보이는 반면 Non-Lambertian은 빛을 특정 방향으로 강하게 반사하여 보는 각도에 따라 표면의 색상이나 밝기가 크게 달라진다.

<img src="/images/paper_review/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis/fig4.png" class="post_img"/>

위 그림은 view dependence 없이 즉, 입력으로 \\(x\\)만 사용했을 때 학습된 모델은 specularities를 잘 표현하지 못하는 어려움이 있다.