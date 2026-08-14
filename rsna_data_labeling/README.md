# RSNA Knee Abnormality Detection — Report Data Labeling

무릎 MRI 판독문(`Report`)을 외부 의학 문헌에서 정리한 질환별 guidance와
오픈소스 LLM을 이용해 12개 타깃으로 라벨링하는 프로젝트입니다.

하나의 Report에 대해 각 타깃을 독립적으로 판정하며,
각 타깃에 대응하는 guidance 문서 전체와 Report를 함께 LLM에 입력합니다.

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

## 처리 흐름

```text
Radiology Report
      +
Target Guidance
      ↓
Mistral-7B-Instruct-v0.3
      ↓
Target Label: 0 / 1
```

Report 하나에 대해 위 과정을 12개 타깃에 각각 수행합니다.

## 프로젝트 구조

```text
rsna_data_labeling/
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
├── kaggle_quickstart.py
├── pyproject.toml
├── requirements-kaggle.txt
└── .gitignore
```

## 주요 파일

### `guidance/`

12개 타깃별 의학 guidance 문서를 저장합니다.

각 문서는 외부의 의학·영상의학 문헌을 기반으로 정리되어 있으며,
해당 질환의 정의, MRI 소견, 감별 시 주의점, 관련 용어 등을 포함합니다.

예:

```text
ACL              -> guidance/acl.md
MCL              -> guidance/mcl.md
Medial Meniscus  -> guidance/medial_meniscus.md
Fracture         -> guidance/fracture.md
```

### `src/knee_guidance/constants.py`

12개 타깃과 각 타깃에 대응하는 guidance 파일명을 정의합니다.

### `src/knee_guidance/guidance.py`

`guidance/` 폴더의 `.md` 파일을 읽어 타깃별 guidance를 제공합니다.

### `src/knee_guidance/prompting.py`

다음 정보를 이용해 타깃별 LLM prompt를 생성합니다.

```text
Target
+
Medical Guidance
+
Radiology Report
```

Report가 영어가 아닌 경우에도 원문의 의학적 의미를 해석하도록 지시하며,
부정 표현, 불확실성, 해부학적 위치와 감별 진단을 고려하도록 구성되어 있습니다.

### `src/knee_guidance/llm.py`

기본 모델은 다음과 같습니다.

```text
mistralai/Mistral-7B-Instruct-v0.3
```

Kaggle GPU 환경에서 실행할 수 있도록 기본적으로 4-bit NF4 양자화를 사용합니다.

모델 출력은 다음 형태의 JSON으로 제한합니다.

```json
{
  "target": "ACL",
  "label": 0
}
```

### `src/knee_guidance/classifier.py`

하나의 Report에 대해 12개 타깃을 순차적으로 판정합니다.

```text
Report + acl.md              -> ACL
Report + mcl.md              -> MCL
Report + medial_meniscus.md  -> Medial Meniscus
...
Report + fracture.md         -> Fracture
```

### `src/knee_guidance/evaluation.py`

Expert label이 존재하는 데이터에 대해 다음 평가 지표를 계산합니다.

- Accuracy
- Precision
- Recall
- F1-score
- Overall Accuracy

## Kaggle 환경 설정

Kaggle에서 저장소를 `/kaggle/working`에 clone한 경우:

```bash
!pip install -q bitsandbytes
!pip install -e /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling --no-deps
```

## Expert label 데이터 검증

전체 실행 전에 일부 데이터만 확인하려면 `--limit`을 사용합니다.

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output-dir /kaggle/working/guidance_validation_test \
  --limit 2
```

58개 expert-labeled Report 전체를 검증하려면:

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output-dir /kaggle/working/guidance_validation
```

실행 후 다음 파일이 생성됩니다.

```text
guidance_validation/
├── predictions.csv
├── metrics_by_target.csv
├── errors.csv
└── raw_outputs.csv
```

### `predictions.csv`

각 Report의 실제 label과 모델 예측값을 함께 저장합니다.

### `metrics_by_target.csv`

12개 타깃별 Accuracy, Precision, Recall, F1-score를 저장합니다.

### `errors.csv`

실제 label과 예측 label이 다른 사례만 저장합니다.

### `raw_outputs.csv`

각 타깃에 대해 Mistral이 생성한 원본 출력을 저장합니다.
JSON parsing 문제나 출력 형식 오류를 확인할 때 사용할 수 있습니다.

## 전체 Report pseudo-label 생성

Expert-labeled 데이터에서 성능을 확인한 뒤,
label이 없는 Report에 대해 pseudo-label을 생성합니다.

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/scripts/label_unlabeled.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection/rsna_data_labeling/guidance \
  --output /kaggle/working/train_pseudo_labels.csv
```

결과는 다음 위치에 저장됩니다.

```text
/kaggle/working/train_pseudo_labels.csv
```

## 실행 결과 활용

생성된 pseudo-label은 이후 MRI 영상 학습 데이터의 target으로 사용할 수 있습니다.

```text
MRI Study / Series
        +
Report 기반 pseudo-label
        ↓
영상 모델 학습
```

먼저 expert-labeled Report에서 타깃별 성능과 오류 유형을 확인한 뒤
전체 Report에 pseudo-label을 적용하는 것을 권장합니다.
