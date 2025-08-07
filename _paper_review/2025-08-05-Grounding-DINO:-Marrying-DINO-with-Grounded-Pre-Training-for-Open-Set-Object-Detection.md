---
title: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
layout: single
collection: paper_review
author_profile: true
---

# Introduction
Artificial General Intelligence(AGI)의 능력을 판단하는 지표 중 하나는 open-world를 얼마나 잘 처리할 수 있는지에 달려있다. 이 논문에서는 사람의 언어 입력에 따라 임의의 object를 탐지할 수 있는 시스템을 개발을 목표로 한다. 이를 `open-set object detection`이라 불린다.

이 모델을 사용하여 이미지 편집을 위한 generative models과 결합하여 동작할 수 있음을 보인다.

<img src="/images/paper_review/Grounding-DINO:-Marrying-DINO-with-Grounded-Pre-Training-for-Open-Set-Object-Detection/fig1-b.png" class="post_img"/>

이를 위해 2가지 원칙에 따라 grounding dino를 설계한다.
1. Tight modality fusion based on DINO
2. Large-scale grounded pre-train for concept generalization

## Tight modality fusion based on DINO

## Lage-scale grounded pre-train for zero-shot transfer


# Related Work
## Detection Transformers

## Open-Set Object Detection

# Grounding DINO
## Feature Extraction and Enhancer

## Language-Guided Query Selection

## Cross-Modality Decoder

## Sub-Sentence Level Text Feature

## Loss Function

# Experiments

## Implementation Details

## Zero-Shot Transfer of Grounding DINO

## Referring Object Detection Settings

## Effects of RefC and COCO Data

## Ablations

# Conclusion