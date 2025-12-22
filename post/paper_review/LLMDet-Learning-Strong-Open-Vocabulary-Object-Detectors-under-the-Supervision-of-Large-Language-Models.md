# Introduction
`Open-vocabulary Object Detection`은 사용자가 입력한 text label을 기반으로 클래스를 탐지하는 task이다. `GLIP`은 object detection과 phrase grounding을 region-word contrastive pre-training을 통해 통합하였다. 이를 통해 학습된 표현은 의미적으로 풍부한 특성을 가진다.

이후 연구들은 vision-language fusion과 정밀한 region-word align에 집중하고 있고, 정교하게 설계된 word embedding이나 negative sample 구성을 도입한다. 또한, pretraining data와 연산 규모를 확장함으로써 open-vocabulary object detector는 다양한 benchmark에서 높은 zero-shot 성능을 보여주고 있다.

최근 연구들은 grounding task를 다른 language task랑 통합하는 것이 시각적인 표현을 언어 지식으로 풍부하게 만들어 보다 강력한 open-vocabulary detector를 구축할 수 있음을 보여주고 있다. `GLIPv2`는 gorunding loss와 masked language modeling(MLM) loss를 함께 활용해 모델을 pretraining 하였다. `CapDet`와 `DetCLIPv3`는 dense captioning과 grounding의 통합이 open-vocabulary 성능 향상에 도움이 됨을 보여줬지만, 각 object에 대해 짧은 caption을 생성하며 이는 대체로 coarse한 설명이나 계층적 class label 수준으로 세분화의 부족, 개별적, object 간 관계 부족이라는 한계를 가진다. 반면, image 전체 수준에서 작성된 길고 상세한 caption은 짧은 region-level보다 더 많은 정보를 담고 있기 때문에 이러한 image-level의 caption이 open-vocabulary detector에 어떤 장점을 가져올 수 있는지 탐구하고 있다.

이를 통해 `LLMDet`는 표준 grounding objective와 함께 caption generation objective을 활용해 open-vocabulary detector를 학습한다. detector에는 large language model(LLM)이 추가되어 LLM은 image feature와 region feature을 입력받아 image-level의 길고 상세한 caption과 각 region에 대한 짧은 phrases를 각각 예측한다. 이를 통해 다음과 같은 네가지 장점을 가진다.

1. **길고 상세한 caption은 각 object에 대해 더 많은 정보를 제공한다.** <br/>
object의 종류, 질감, 색상, 부분 요소, 동작, 위치, text 등의 정보를 담은 자세한 caption은 더욱 풍부한 vision-labnguage 표현을 학습하는데 유리하지만, region-level caption은 지나치게 단순하다.
2. **Image-level caption은 image 전체 요소를 통합하여 정렬한다.** <br/>
이 방식은 foreground object뿐만 아니라 background와 object 간 관계까지 모델링하며, 단순히 region of interest에만 집중하는 방식보다 더 많은 정보와 포괄적인 이미지 이해를 제공한다.
3. **Image-level caption은 region-level annotation보다 확장성이 뛰어난다.** <br/>
최신 large vision-language model은 전체 이미지 이해에는 강하지만, 정밀한 region-level 이해에는 어려움이 있다. 적절한 prompt를 사용하면 고품질의 image caption을 저비용으로 생성할 수 있다.
4. **Fully-pretrained LLM은 본질적으로 open-vocabulary의 특성을 가진다.** <br/>
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

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/figure2.png)

LLMDet를 grounding loss와 captionng loss로 학습할 수 있도록 하기 위해 각 training sample을 4개의 요소로 구성된 $(I, T_g, B, T_c)$로 정의한다. 각 요소는 다음과 같다.

- $I$ : image
- $T_g$ : short grounding text
- $B$ : annotated bounding boxes mapped by phrase in the grounding text
- $T_c$ : detailed caption for the whole image

이때 전체 이미지를 위한 detailed caption ($T_c)$을 수집할 때, 다음 2가지 원칙을 따른다.

1. caption은 가능한 많은 detail을 포함해야 한다.<br\>
caption이 object의 종류, 질감, 색상, object의 부분, object의 동작, 정확한 위치, 이미지 내 text 등을 묘사하여 정보가 풍부하길 기대한다.
2. caption은 image에 대한 사실 기반의 detail만 포함해야 한다. <br/>
상상에 기반한 설명이나 추론 중심의 caption이 너무 많을 경우 정보 밀도가 떨어지거나 model 학습에 방해가 될 수 있다. 따라서 사실 기반이면서 정보 밀도가 높은 caption을 추구하며 학습 효율을 놓인다.

## Dataset Construction
Data 구축 비용을 절감하기 위해 기존에 존재하는 bounding box 또는 detailed caption을 보유한 dataset을 기반으로 작업한다. 기존 연구를 통해 object detection dataset, grounding dataset, image-text dataset에서 데이터를 수집한다.

- **Object detection dataset**
    - COCO와 V3Det을 사용
    - ShareGPT4V에서 168k개의 detailed caption을 수집
    - ASv2로부터 42k개의 caption을 수집하였으며, object 간 관계에 집중된 caption
    - V3Det는 13,000개 이상의 category를 보유하고 있어, vocabulary를 크게 향상 가능
        - V3Det의 caption은 Qwen2-VL-72b를 활용해 직접 생성하였으며, prompt는 위 그림과 같이 제시되었다.
    - GLIP을 따르며, detection dataset의 grounding text는 클래스 이름들을 연결한 문자열로 구성된다. (ex. "char. fork. cup. cow")
- **Grounding dataset** <br/>
널리 사용되는 GoldG를 채택하였으며, GQA, Flickr30k를 포함한다. 이때, 원본 annotation에는 각 image에 대해 짧은 grounding text들이 다수 존재하지만, 효율적으로 처리하고 negative sample을 늘리기 위해, bounding box 충돌이 없는 gorunding text들을 간단한 문자열 연결 방식으로 하나로 병합했으며, 이로 인해 dataset은 769k에서 437k로 downsampling이 되었다. 또한, detailed caption은 Qwen2-VL-72b를 통해 직접 생성하였다.
- **Image-text dataset** <br/>
LCS-558k를 사용했으며, 이는 LLaVA-OneVision과 ShareGPT4v로부터 detailed caption이 포함되어 있다. 이 dataset에 대한 pseudo box를 생성하기 위해 caption에서 noun phrases를 language parser을 이용해 추출한 후, MM Grounding DINO(Swin-L)을 사용해 각 phrase에 대한 bounding box를 생성했다. 3개 미만의 bounding box만 존재하는 image는 제거하였고, grounding text는 detection dataset과 마찬가지로 phrase를 연결하여 구성했다.

최종 dataset인 `GroundingCap-1M`은 총 112만(1120k)개의 sample을 포함하였다.

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table1.png)

## Quality Verification
data 수집 과정에서 prompt를 신중히 설계하고 접근 가능한 모델인 `Qwen2VL-72b`를 사용했다고 한다. 이 model은 학습 과정에서 hallucination을 방지하기 위해 많은 노력이 들어갔다고 한다. 하지만 여전히 noise가 있어 이를 처리하기 위해 후처리 과정을 거쳤다.
1. 추측성 문구 제거 <br/>
model이 상상이나 추론을 피하도록 prompt를 제공했음에도 불구하고, "indication", "suggesting", "possibly" 등과 같은 단어로 추론을 포함하는 경우가 많기 때문에 이런 단어가 포함된 문장을 제거한다.
2. 무의미한 caption 제거 <br/>
"In the image, a man a man a man..."(반복되는 문장) 혹은 "Sorry, I can not answer the question."과 같은 문장을 거러내기 위해 규칙 기반 filtering을 설계했다.
3. Detail 강화
caption이 100 token 미만인 이미지에 대해서는, Qwen2VL-72b를 이용해 재생성을 수행하여 detail을 보완했다. 이러한 이중 검증 mechanism을 통해 데이터 품질을 확보했으며, 후처리 이후 각 caption은 평균적으로 약 115 단어를 포함하게 되었다. 정량적 분석은 [Section 5.3](#ablation-study)에서 확인해 볼 수 있다.

# Training LLMDet under the Supervision of Large Language Models
Grounding task와 다른 language 관련 task들을 통합하면, vision features을 언어 지식으로 더욱 풍부하게 만들 수 있고, 이로 인해 visiual 개념을 확장하고, vision-language alignment 성능을 향상시킬 수 있다. 기존 연구들은 주로 dense captioning에 초점을 맞추어, language model이 region of interest을 짧게 설명하는 caption 또는 class 이름을 생성하는 방식으로 설계되어 왔다. 하지만 이는 개별 object에 대한 상세한 정보, object 간의 관계, foreground 및 background에 대한 정보 등을 간과한다. 이러한 정보는 하나의 image-level caption 안에 모두 포함될 수 있다.

해당 논문에서 region-level의 open-vocabulary object detector 또한, large language model(LLM)의 supervision 하에 생성된 긴 image caption으로부터 이점을 얻을 수 있다.

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/figure3.png)

먼저 pretrained DETR 기반 open-vocabulary object detector와 LLM을 사용하여 caption을 생성한다. 하지만 두 모델은 별도로 pretraining 되었기 때문에 detector에서 나온 vision feature을 LLM의 입력 공간으로 mapping 해주는 projector를 학습한다. (Step1)

- DETR의 encoder에서 나오는 p5 feature map을 LLM의 입력으로 사용하고, LLM은 이 feature을 바탕으로 전체 이미지에 대한 caption을 생성한다.
- 이 과정에서는 language modeling loss를 통해 LLM을 supervision하며, 오직 projector만 학습이 가능하게 설정한다.

pre-alignment 후, detector, projector, LLM을 end-to-end 방식으로 finetuning한다. (Step2)
기존의 grounding task (ex. word-region alignment loss $(L_{align})$, box regression loss $(L_{box})$) 외에도 다음 2가지 task가 추가된다.

1. Image-level caption generation
2. Region-level caption generation

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/figure4.png)

## Image-level caption generation
이 task에서는, language model이 detector로부터 나온 feature map을 입력으로 받아, GroundingCap-1M에서 annotation된 긴 detailed caption을 출력한다. 이때, multi-modal 훈련의 일반적인 방식에 따라, LLM의 입력을 conversation format으로 구성한다. 구성 방식은 다음과 같다.

- System message
- User Input: detector로부터 얻은 vision feature와 prompt(ex. "Describe the image in detail.")
- Answer: Grounding Cap-1M의 caption

이때, LLM은 user input에 기반하여 caption을 새엇ㅇ하며, 이는 language modeling loss $(L_{lm}^{image})$로 학습된다. 이 과정에서 생성되는 caption은 다양한 visual detail과 image에 대한 포괄적인 이해를 포함하므로, detector feature에 잘 표현되어야 LLM이 손실을 줄이고 정확한 caption을 생성할 수 있다.

## Region-level caption generation
Image-level caption의 경우, LLM은 전체 feature map을 입력으로 받기 때문에, caption 내의 entity(ex. dishes)을 이미지 내의 특정 영역에 정확히 mapping하는 것이 어렵다. 예를 들어 figure2에서 "dish"는 이미지의 아주 작은 부분을 차지해, 유사한 object가 여러개 존재하는 것을 확인해 볼 수 있다. 이를 보완하기 위해 region-level caption generation task을 도입한다.
이 task에서는 detector에서 나온 positive object queries(GT box와 매칭된 query들)를 선택한다. 각각 query에 대해 LLM이 대응되는 grounding phrase를 생성한다.(ex. "young man", "mothor", "dishes") 입력 형식은 image-level task와 유사하게 conversation format을 구성되며, prompt는 "Describe the region in a phrase."로 구성한다. 이때, 각 query의 feature는 제한적이므로, LLM에 cross-attention layer를 추가하여 detector의 전체 feature map으로부터 필요한 정보를 추출한다. 이때, image-level caption generation에서 사용되는 vision/text token은 이 cross-attention layer을 통과하지 않는다. LLM이 object query에 대해 정확한 phrase를 출력함으로써, image 내의 entity를 특정 region에 정확하게 mapping할 수 있다.

최종 loss는 다음과 같다.

$$
L = L_{align} + L_{box} + L^{image}_{lm} + L^{region}_{lm}
$$


# Experiment
## Implementation Details
해당 녕구에서 MM Grounding DINO를 baseline으로 선택하였다. pretrained checkpoint를 그대로 load한 후, GroundingCap-1M dataset을 이용해 grounding loss와 caption generation loss를 동시에 이용해 finetuning을 한다. GroundingCap-1M에 포함된 image의 상당수가 MM-GDINO의 pretraining dataset(ex. GoldG, V3Det)과 중복된다. 이때, MM-GDINO는 완전히 pretrin된 상태이므로, vision backbone은 학습 중 freeze 상태로 유지한다. 사용된 large language model은 `LLaVA-OneVision-0.5b-ov`에서 초기화한다. 메모리 절약과 학습 효율 향상을 위해 image-level caption generation을 위한 최대 token 길이는 1600, region-level generation을 위한 token 길이는 40으로 설정한다. 이미지당 caption generation을 위한 최대 region의 개수는 16개로 제한한다.

Image-level visual input은 detector의 encoder에서 추출된 p4 및 p5 feature map을 사용한다. p4는 27x27, p5는 20x20으로 resize하고, 이 둘을 하나의 token sequence로 결합한다. 전체 학습은 약 150,000 iteration (2epoch)동안 수행되며, batch size는 16이다. 이는 NVIDIA L20 GPU 8개로 약 이틀정도 소요된다.

## Zero-Shot Detection Transfer Ability
사용한 benchmark는 다음과 같다.

- LVIS
- ODinW13/35
- COCO-O
- RefCOCO
- RefCOCO+
- RefCOCOg
- gRefCOCO

훈련 시 COCO data를 사용하기 때문에 RefCOCO 계열 dataset의 validation/test image는 MM-GDINO의 설정에 따라 GroundingCap-1M에서 제거했다. LVIS minival image들은 COCO train set과 겹치지 않기 때문에, 엄밀한 zero-shot 설정을 만족한다. test시에는 LLM을 제거하 inference cost는 baseline(MM-GDINO)와 동일하다.

### Zero-shot performance on LVIS
LVIS는 총 1203개 class를 포함한 detection dataset이다. 이 class는 등장 빈도에 따라 `frequent`/`common`/`rare` 3가지로 나뉜다. 전체 class는 40개씩 31개 chunck로 분할하여, 각 Image는 31번 추론된다.

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table2.png)

Table 2에 따르면, 새로운 학습 objective 및 dataset을 사용한 LLMDet는 MM-GDINO 대비 LVIS minival에서 각각 $3.3%/3.8%/14.3% AP$, $3.1%/3.3%/17.0% AP_r$ 향상을 보인다. 이때, MM-GDINO에서 Swin-L backbone을 사용할 경우, 성능이 매우 낮은데 이는 pretraining dataset(V3Det 부족 등)의 차이 때문이라고 추정한다. 반면, 같은 Swin-L backbone을 사용한 LLMDet는 훨씬 적은 학습 데이터로도 다른 SOTA 기법을 능가하여 50.6% AP를 기록하였다.

DetCLIP series는 class 간 균형 잡힌 성능을 보이는데, 이는 정제된 dataset과 잘 구성된 명사 개념 집합 덕분디ㅏ. LLMDet 또한 DetCLIP에 적용 가능할 것이라 보인다.

### Zero-shot performance on ODinW

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table3.png)

ODinW (Object Detection in the Wild)는 다양한 domain과 vocabulary를 아우르는 35개의 dataset을 포함하고 있으며, open-vocabulary 성능을 평가하는데 적합하다. 기존 연구에 따라, 13개 선택 dataset(ODinW13) 및 35개 전체 dataset(ODinW35)에 대한 average AP를 보고한다. LLMDet는 ODinW35에서 최고 AP를 달성하며, 다양한 domain 전이에 뛰어나 능력을 보인다.

### Zero-shot performance on COCO-O

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table4.png)

COCO-O는 COCO와 동일한 80개 class를 포함하되, 도메인이 완전히 다르다 (ex. sketch, weathre, cartoon, painting, tattoo, handmake). LLMDet는 MM-GDINO 대비 2.1% AP 향상을 보여 domain 변화에 강한 견고성을 입증하였다.

### Zero-shot performance on Referring Expression Comprehension (REC)

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table5.png)

REC는 문구로 언급된 object를 찾아내는 task로, 언어 이해 및 정밀한 vision-language align이 필요하다. LLM과 함께 정밀한 caption을 co-trianing한 LLMDet는 더 풍부한 시각 표현 학습이 가능하며, 결과적으로 다양한 REC dataset에서 MM-GDINO 대비 향상된 성능을 기록했습니다.

## Ablation Study
해당 연구는 Swin-T backbone을 기준으로 실험을 수행하며, LVIS minival에서 성능을 보고한다.

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table6.png)

### Effect of the main components of LLMDet
grounding annotation만 사용했을 때는 41.4% $\left$ 43.8% AP의 성능 향상이 있고, region-level caption만 사용했을때는 성능 향상이 없었다. image-level caption만 사용했을 때는 약간의 향상이 있었다. 이 이유를 LLM이 전체 이미지에서 특정 object를 mapping하는데 어려움이 존재한 것으로 보인다. 따라서 region-level과 image-level을 모두 사용하는 것이 가장 큰 효과를 발휘한다. 

### Effect of different large language models

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table7.png)

기본 LLM으로는 LLaVA-OneVision-0.5b-ov (Qwen2-0.5b-instruct 기반)을 사용한다. 다른 vision encoder를 사용했음에도 불구하고 multi modal pretraining이 rare class에 $+2.2% AP_r$ 향상을 유도했다. 하지만 LLM 사이즈 증가는 성능 향상에 미미한 영향을 주었는데, 이는 추론 능력은 향상되지만 시각 표현 학습에는 크게 도움이 되지 않기 때문이라고 분석된다.

### Effect of generated captions’ quality

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table8.png)

Table 8에 따르면, Qwen2VL-72b 대신 LLaVA-OneVision-7B caption을 사용했을 때, $0.8% AP, 4.1% AP_r$ 감소되었고, COCO caption, LLaVA의 LCS caption, GoldG의 short phrase을 사용했을 때, $0.4% AP$ 감소했다. 각 설정에서 300개의 caption-image 쌍을 무작위로 sampling하고, GPT-4o로 caption의 상세도와 hallucination을 평가했다. 결과적으로 GroundingCap-1M caption이 가장 상세하고 적당한 hallucination 수준을 보였으며 이는 dataset 품질이 우수함을 보였다.

### Effect of the pretraining data

![](/images/paper_review/LLMDet-Learning-Strong-Open-Vocabulary-Object-Detectors-under-the-Supervision-of-Large-Language-Models/table8.png)

GroundingCap-1M은 계산 자원의 제한으로 인해 100만 sample만 포함된다. Table 9에 따르면 LCS 제외(813k data)를 한 data에 따라 $42.8% AP$로 하락하였다. 이를 보아 데이터가 많을수록 성능 향상이 가능하다고 보인다. 또한 image-level caption에서 추측성 표현을 제거하지 않으면 성능이 $44.2% AP, 35.0% AP_r$로 하락하여, hallucination이 rare class 성능에 부정적 영향을 미치는 것을 확인할 수 있다.

### Effect of the cross-attention layers in LLM
region-level generation에서는 object query가 cross-attention을 통해 encoder feature map에서 정보를 얻지 않으면 $44.0% AP$로 하락됨을 확인할 수 있다. 반면, image-level generation에서는 전체 feature map 사용때문에 cross-attention이 도움이 되지 않는다.

### Effect of pretraining the projector before end-to-end finetuning
LLM과 detector는 개별적으로 pretraining 되기 때문에 projector를 pretraining하여 feature space를 정렬시키는 것이 중요하다. 만약 pretraining을 생략하면 rare class AP가 3.5% 감소하며, frequent class는 annotation이 많이 영향이 적은 것을 확인할 수 있다.

# Conclusion
본 연구에서는 기존의 open-vocabulary detector들의 성능을 향상시키기 위한 새로운 학습 objective를 탐구gksek. 해당 연구에서는 **대형 언어 모델(LLM)**을 활용하여 다음 두 가지를 생성한다.

- image-level detailed captions
- region-level coarse grounding phrases

이러한 방식으로, **detector**는 상세한 캡션으로부터 더 많은 정보와 이미지에 대한 포괄적인 이해를 얻게 되며, 결과적으로 **풍부한 vision-language 표현**을 구축할 수 있다. 이러한 방식으로 구축된 탐지기인 LLMDet는 다양한 benchmark에서 SOTA 성능을 달성했다. 또한, 향상된 LLMDet는 반대로 **강력한 대규모 multimodal model**을 구축하는 데에도 기여할 수 있어, **mutual benefits**이 가능함을 보여준다.