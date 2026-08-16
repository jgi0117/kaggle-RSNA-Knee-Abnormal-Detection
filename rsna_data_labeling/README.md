# RSNA Knee Abnormality Detection — Report Data Labeling

무릎 MRI 판독문(`Report`)을 타깃별 medical guidance와 오픈소스 LLM을 이용해
12개 이상 소견으로 pseudo-labeling 하는 프로젝트입니다.

현재 평가 파이프라인은 **기존 번역 결과를 재사용한 뒤 Qwen3-14B로 12개 타깃을 분류**하도록 구성되어 있습니다.

## 라벨링 대상

```text
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
```

## 현재 처리 흐름

```text
Original Radiology Report
        ↓
Qwen3-8B translation
        ↓
English translated_report
        +
Target-specific Guidance
        ↓
Qwen3-14B 4-bit NF4
        ↓
Target Label: 0 / 1
```

Report 하나에 대해 번역은 한 번만 수행하고,
분류는 12개 타깃 각각에 대해 독립적으로 수행합니다.

`rsna_data_labeling/data/translations.csv`가 존재하면
기존 번역 결과를 자동으로 재사용하여 Qwen translation을 건너뜁니다.

## 프로젝트 구조

```text
rsna_data_labeling/
├── data/
│   └── translations.csv
│
├── guidance/
│   ├── acl.md
│   ├── mcl.md
│   ├── medial_meniscus.md
│   ├── lateral_meniscus.md
│   ├── medial_oa.md
│   ├── lateral_oa.md
│   ├── pf_oa.md
│   ├── effusion.md
│   ├── synovitis.md
│   ├── bakers.md
│   ├── contusion.md
│   └── fracture.md
│
├── scripts/
│   ├── validate_expert.py
│   └── label_unlabeled.py
│
├── src/
│   └── knee_guidance/
│       ├── __init__.py
│       ├── classifier.py
│       ├── constants.py
│       ├── evaluation.py
│       ├── guidance.py
│       ├── llm.py
│       └── prompting.py
│
├── pyproject.toml
├── requirements-kaggle.txt
└── README.md
```

## 주요 파일

### `data/translations.csv`

번역 결과를 재사용하기 위한 캐시 파일입니다.

필수 컬럼:

```text
index
translated_report
```

선택적으로 `original_report` 컬럼을 포함할 수 있습니다.

Expert validation에서는 필요한 expert index가 모두 존재해야 하며,
중복 index 또는 비어 있는 `translated_report`가 있으면 실행을 중단합니다.

### `guidance/`

12개 타깃별 medical guidance 문서를 저장합니다.

각 target은 자신의 guidance와 translated report만 이용해 독립적으로 판정합니다.

### `src/knee_guidance/constants.py`

12개 target 이름과 각 guidance 파일명을 정의합니다.

### `src/knee_guidance/guidance.py`

`guidance/` 폴더의 `.md` 파일을 읽어 target별 guidance를 제공합니다.

### `src/knee_guidance/prompting.py`

두 종류의 prompt를 생성합니다.

```text
Translation prompt
Original Report
→ English translated_report
```

```text
Classification prompt
Target
+ Target Guidance
+ translated_report
→ binary label
```

### `src/knee_guidance/llm.py`

현재 classification 기본 모델:

```text
Qwen/Qwen3-14B
```

Kaggle GPU 환경에서 실행할 수 있도록 기본적으로 **4-bit NF4** 양자화를 사용합니다.

현재 설정:

```text
Model                 Qwen/Qwen3-14B
Quantization          4-bit NF4
Compute dtype         float16
Double quantization   enabled
Thinking mode         disabled
Classification output JSON
```

번역이 새로 필요한 경우 기본 translator는:

```text
Qwen/Qwen3-8B
```

입니다.

분류 결과는 다음 형태의 JSON으로 요청합니다.

```json
{
  "target": "ACL",
  "label": 0
}
```

### `src/knee_guidance/classifier.py`

하나의 translated report에 대해 12개 target을 순차적으로 판정합니다.

```text
translated_report + acl.md              → ACL
translated_report + mcl.md              → MCL
translated_report + medial_meniscus.md  → Medial Meniscus
...
translated_report + fracture.md         → Fracture
```

한 target에서 generation 또는 parsing 오류가 발생해도
나머지 target은 계속 처리하며 실패 내역을 별도로 저장합니다.

### `src/knee_guidance/evaluation.py`

Expert label이 존재하는 데이터에 대해 다음 평가 지표를 계산합니다.

- Accuracy
- Precision
- Recall
- F1-score
- Overall Accuracy

## Kaggle 환경 설정

저장소를 `/kaggle/working`에 clone한 경우:

```bash
!pip install -q -U \
  "transformers>=4.51.0,<5" \
  "accelerate>=1.2.0" \
  "bitsandbytes>=0.45.0" \
  "huggingface_hub>=0.27.0"

!pip install -e \
  /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling \
  --no-deps
```

설치 후 이미 `transformers` 또는 `bitsandbytes`를 import한 상태였다면
Kaggle kernel을 한 번 재시작하는 것을 권장합니다.

GPU 환경은 다음 코드로 확인할 수 있습니다.

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
```

## Expert label 데이터 검증

전체 실행 전에 일부 데이터만 확인하려면 `--limit`을 사용합니다.

```bash
!python \
/kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output-dir /kaggle/working/qwen3_14b_validation_test \
  --limit 2
```

58개 expert-labeled Report 전체를 검증하려면:

```bash
!python \
/kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output-dir /kaggle/working/qwen3_14b_validation
```

classifier model을 명시하고 싶다면:

```bash
!python \
/kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output-dir /kaggle/working/qwen3_14b_validation \
  --classifier-model Qwen/Qwen3-14B
```

기존 translation 파일이 기본 위치가 아닌 경우:

```bash
--translations-csv /path/to/translations.csv
```

를 추가합니다.

기본 위치는:

```text
rsna_data_labeling/data/translations.csv
```

입니다.

## Validation 출력

실행 후 다음 파일이 생성됩니다.

```text
qwen3_14b_validation/
├── predictions.csv
├── metrics_by_target.csv
├── errors.csv
├── raw_outputs.csv
├── failures.csv
├── translations.csv
├── predictions_checkpoint.csv
├── raw_outputs_checkpoint.csv
└── failures_checkpoint.csv
```

### `predictions.csv`

각 Report의 expert label과 Qwen3-14B prediction을 함께 저장합니다.

### `metrics_by_target.csv`

12개 target별 Accuracy, Precision, Recall, F1-score를 저장합니다.

### `errors.csv`

expert label과 prediction이 다른 사례만 저장합니다.

### `raw_outputs.csv`

각 target에 대한 LLM 원본 출력을 저장합니다.

JSON parsing 또는 generation 동작을 확인할 때 사용할 수 있습니다.

### `failures.csv`

generation 또는 parsing에 실패한 target을 저장합니다.

실패 prediction은 evaluation에서 제외되므로,
모델 성능을 비교할 때 반드시 `failures.csv`와 failure count를 함께 확인해야 합니다.

### checkpoint 파일

각 study 처리가 끝날 때 현재까지의 결과를 저장합니다.

따라서 중간에 Kaggle runtime이 종료되어도
완료된 study의 결과를 확인할 수 있습니다.

## 전체 Report pseudo-label 생성

Expert validation에서 성능을 확인한 뒤
label이 없는 Report에 대해 pseudo-label을 생성합니다.

```bash
!python \
/kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/label_unlabeled.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output /kaggle/working/train_pseudo_labels_qwen3_14b.csv
```

`translations.csv`에 해당 index가 있으면 기존 번역을 사용합니다.

해당 index의 번역이 없으면 Qwen3-8B를 로드하여
누락된 Report만 영어로 번역한 뒤 Qwen3-14B classification을 수행합니다.

## 모델 비교 원칙

Qwen3-14B 실험에서는 이전 Qwen3-8B 결과와 직접 비교하기 위해
가능한 한 다음 조건을 동일하게 유지합니다.

```text
translations.csv
target-specific guidance
classification prompt
expert validation data
evaluation metrics
quantization method
thinking disabled
```

주요 변경 변수는 classification model입니다.

```text
Qwen3-8B
   ↓
Qwen3-14B
```

따라서 Qwen3-14B 결과는 기존 Qwen3-8B baseline과
동일한 조건에서 모델 크기 증가 효과를 비교하는 용도로 사용합니다.

## 실행 결과 활용

최종 pseudo-label은 이후 MRI 영상 학습 데이터의 target으로 사용할 수 있습니다.

```text
MRI Study / Series
        +
Report 기반 pseudo-label
        ↓
영상 모델 학습
```

먼저 expert-labeled Report에서 target별 성능과 오류 유형을 확인한 뒤
전체 Report에 pseudo-label을 적용하는 것을 권장합니다.
