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
