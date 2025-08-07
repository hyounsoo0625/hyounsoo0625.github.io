# Introduction
`Open-vocabulary Object Detection`은 사용자가 입력한 text label을 기반으로 클래스를 탐지하는 task이다. `GLIP`은 object detection과 phrase grounding을 region-word contrastive pre-training을 통해 통합하였다. 이를 통해 학습된 표현은 의미적으로 풍부한 특성을 가진다.

이후 연구들은 vision-language fusion과 정밀한 region-word align에 집중하고 있고, 정교하게 설계된 word embedding이나 negative sample 구성을 도입한다. 또한, pretraining data와 연산 규모를 확장함으로써 open-vocabulary object detector는 다양한 benchmark에서 높은 zero-shot 성능을 보여주고 있다.

최근 연구들은 grounding task를 다른 language task랑 통합하는 것이 시각적인 표현을 언어 지식으로 풍부하게 만들어 보다 강력한 open-vocabulary detector를 구축할 수 있음을 보여주고 있다. `GLIPv2`는 gorunding loss와 masked language modeling(MLM) loss를 함께 활용해 모델을 pretraining 하였다. `CapDet`와 `DetCLIPv3`는 dense captioning과 grounding의 통합이 open-vocabulary 성능 향상에 도움이 됨을 보여줬지만, 각 object에 대해 짧은 caption을 생성하며 이는 대체로 coarse한 설명이나 계층적 class label 수준으로 세분화의 부족, 개별적, object 간 관계 부족이라는 한계를 가진다. 반면, image 전체 수준에서 작성된 길고 상세한 caption은 짧은 region-level보다 더 많은 정보를 담고 있기 때문에 이러한 image-level의 caption이 open-vocabulary detector에 어떤 장점을 가져올 수 있는지 탐구하고 있다.

이를 통해 `LLMDet`는 표준 grounding objective와 함께 caption generation objective을 활용해 open-vocabulary detector를 학습한다. detector에는 large language model(LLM)이 추가되어 LLM은 image feature와 region feature을 입력받아 image-level의 길고 상세한 caption과 각 region에 대한 짧은 phrases를 각각 예측한다. 이를 통해 다음과 같은 네가지 장점을 가진다.

1. **길고 상세한 caption은 각 object에 대해 더 많은 정보를 제공한다.**<br/>
object의 종류, 질감, 색상, 부분 요소, 동작, 위치, text 등의 정보를 담은 자세한 caption은 더욱 풍부한 vision-labnguage 표현을 학습하는데 유리하지만, region-level caption은 지나치게 단순하다.
2. **Image-level caption은 image 전체 요소를 통합하여 정렬한다.**<br/>
이 방식은 foreground object뿐만 아니라 background와 object 간 관계까지 모델링하며, 단순히 region of interest에만 집중하는 방식보다 더 많은 정보와 포괄적인 이미지 이해를 제공한다.
3. **Image-level caption은 region-level annotation보다 확장성이 뛰어난다.**<br/>
최신 large vision-language model은 전체 이미지 이해에는 강하지만, 정밀한 region-level 이해에는 어려움이 있다. 적절한 prompt를 사용하면 고품질의 image caption을 저비용으로 생성할 수 있다.
4. **Fully-pretrained LLM은 본질적으로 open-vocabulary의 특성을 가진다.**<br/>
LLM을 이용한 caption 생성을 통해 detector는 자연스롭게 LLM과 alignment되고, 이로 인해 강력한 일반화 성능과 rare class에 대한 성능 향상을 얻게 된다.

기존의 grounding dataset은 image 전체에 대한 상세한 caption을 제공하지 않는다. 이에 따라 이 논문에서는 새로운 dataset인 GroundingCap-1M을 구축하 학습을 수행한다고 한다. 이는 4-tuple로 구성되어 있다고 한다.

- Image
- Short grounding text
- Phrases in the grounding text에 매핑된 annotated bounding boxes
- Long detailed image-level caption

LLM은 region feature와 image feature을 이해하고 각 object에 대해 grounding phrase와 image 전체에 대한 caption을 생성한다. 이 논문에서는 LLM을 기존 detector와 alignment 시킨 후, 전체를 fine-tuning을 한다.

이를 통해 vision foundation model이 LLM의 supervision으로부터 혜택을 받을 수 있음을 입증한다. 이는 단순히 LLM이 생성한 caption을 label로 사용하는 것 뿐만 아니라 LLM과 co-training 과정에서 발생하는 gradient로부터도 영향을 받는다.

이를 통해 향상된 LLMDet를 Large Language Model과 통합하면, 더 강력한 Large Multimodal Model(LMM)을 구축할 수 있다. LLM의 supervision 하에 학습된 LLMDet는 더 강력한 open-vocabulary 능력을 갖추게 될 뿐 아니라 LLM과의 pre-alignment도 수행된다. 따라서, pretraining 된 LLMDet는 강력한 vision foundation model로 활용될 수 있으며, 더 나은 LMM을 만드는데 기여할 수 있게 되녀 mutual benefit을 얻을 수 있다고 한다. [`Appendix`](#Appendix)

# Related Work
## Open-Vocabulary Object Detection
Open-vocabulary object detection (OVD)에서는 detector는 제한된 training dataset으로 학습되지만, test 시에는 임의의 user-input classes을 탐지하는 것을 목표로 한다. 임의의 class을 탐지하기 위해, OVD는 vision-language task로 정식화되며, 이를 통해 detector는 이전에 본적 없는 class도 class의 이름만으로 탐지할 수 있게 된다.

이는 CLIP과 같은 vision-language model의 zero-shot 성능에 영감을 받아 시작되었다고 한다. 이를 통해 detector을 CLIP에 align을 시키는 방식 또는 CLIP을 model의 일부로 통합하는 방식은 OVD 문제를 해결하기 위한 직관적인 접근 방식이다.

하지만, CLIP은 image-level objective로 pretraining되었기 때문에, 그 feature는 OVD에 완벽하게 적합하지 않다. 이에 대한 대안으로 다양한 자원으로부터 대규모 데이터를 수집하여 object-aware한 visual-language space를 구축하는 방식이 있다. 다양한 자원은 다음과 같다.

- image classification datasets
- object detection datasets
- grounding datasets
- image-text datasets

또한, 다른 language task들과의 multi-task learning(ex. masked language modeling, dense captioning)은 vision-language alignment 성능을 향상시켜 detector의 open-vocabulary 성능을 개선할 수 있다. 하지만 기존 연구들은 대부분 region of interest에 대한 short phrases 생성에만 초점을 맞추고 있다. 이에 따라, co-training 과제로서, image-level detailed captions을 large language models을 활용해 생성하는 새로운 방식을 탐색한다.

## Large Vision-Language Model
최근의 large vision-language models는 large language models에 우수한 visual perception과 이해 능력을 더한다. 일반적인 large vision-language model은 다음 3가지 구성 요소로 이루어진다.

1. Vision foundation model: vision token 추출
2. Projector: vision feature을 language space로 mapping
3. Large language model: 시각 정보와 텍스트 정보를 모두 이해

최근 연구에서는 더 나은 vision encoder가 최종 multi-modal 성능을 향상시킨다는 사실을 발견했지만, `Large language model이 vision encoder을 향상시킬 수 있는가?`에는 많이 탐구되지 않았다.

InternVL은 CLIP과 유사한 vision encoder를 6B parameter 규모로 확장하고, text encoder로 large language model을 사용하는 방식을 채택했다.

해당 연구에서는 detector도 large language model로부터 혜택을 받을 수 있으며, 향상된 detector는 다시 large language model의 multi-modal 성능을 향상시킬 수 있음을 보인다.

더 나은 large vision-language model을 학습하기 위해서는 고품질의 caption data가 필수적이다. 특히 LLM의 supervision 하에 open-vocabulary detector를 학습할 때, caption의 품질이 핵심 요소라고 주장한다. 따라서, 기존의 고품질 caption dataset을 적극 활용하고, large vision-language model이 고품질 데이터를 생성하도록 유도한다.

# GroundingCap-1M Dataset
## Data Formulation
<img src="/images/paper_review/LLMDet:-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/figure2.png" class="post_img"/>
LLMDet를 grounding loss와 captionng loss로 학습할 수 있도록 하기 위해 각 training sample을 4개의 요소로 구성된 \\((I, T_g, B, T_c\\)로 정의한다. 각 요소는 다음과 같다.

- \\(I\\) : image
- \\(T_g\\) : short grounding text
- \\(B\\) : annotated bounding boxes mapped by phrase in the grounding text
- \\(T_c\\) : detailed caption for the whole image


# Training LLMDet under the Supervision of Large Language Models

# Experiment
## Implementation Details

## Zero-Shot Detection Transfer Ability

## Ablation Study

# Conclusion