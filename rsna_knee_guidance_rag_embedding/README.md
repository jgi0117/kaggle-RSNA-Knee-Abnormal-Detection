# RSNA Knee Guidance RAG - Embedding Retrieval

Kaggle용 의료 guidance RAG 파이프라인입니다.

- 생성 모델: `mistralai/Mistral-7B-Instruct-v0.3` 4-bit
- 임베딩 모델: `intfloat/multilingual-e5-small`
- 검색: target-aware dense retrieval (cosine similarity)
- 입력: knee MRI `Report`
- 출력: 12개 binary label
- 검증: expert label이 존재하는 58개 row의 Accuracy / Precision / Recall / F1
- 이후: 동일 RAG로 unlabeled report pseudo-label 생성

## 왜 TF-IDF 대신 Embedding Retrieval인가?

Report가 영어뿐 아니라 스페인어, 프랑스어, 독일어, 네덜란드어, 그리스어,
불가리아어, 터키어 등 다양한 언어로 작성되어 있으므로,
단어가 정확히 겹쳐야 잘 동작하는 TF-IDF보다 multilingual embedding 검색이 더 적합합니다.

`multilingual-e5-small`은 query/document를 같은 벡터 공간으로 매핑하므로
예를 들어 스페인어 report의 `rotura del LCA`와 영어 guidance의
`anterior cruciate ligament tear`를 의미 기반으로 연결할 수 있습니다.

## 디렉토리

```text
rsna_knee_guidance_rag_embedding/
├─ guidance/
│  ├─ acl/
│  ├─ mcl/
│  ├─ medial_meniscus/
│  ├─ lateral_meniscus/
│  ├─ medial_oa/
│  ├─ lateral_oa/
│  ├─ pf_oa/
│  ├─ effusion/
│  ├─ synovitis/
│  ├─ bakers/
│  ├─ contusion/
│  └─ fracture/
├─ scripts/
│  ├─ build_index.py
│  ├─ inspect_retrieval.py
│  ├─ validate_expert.py
│  └─ label_unlabeled.py
└─ src/knee_rag/
```

## 1. Kaggle 설치

Kaggle 기본 `torch`, `transformers`, `pandas`, `sklearn`을 사용합니다.
Mistral 4-bit 로딩을 위해 `bitsandbytes`만 추가 설치합니다.

```bash
!pip install -q bitsandbytes
!pip install -e /kaggle/working/rsna_knee_guidance_rag_embedding --no-deps
```

## 2. Guidance 문서 넣기

각 target 폴더 안에 `.md` 또는 `.txt` 형태로 의학 guidance를 넣습니다.

예:

```text
guidance/acl/acr_acl.md
guidance/acl/review_acl.md
guidance/pf_oa/pf_oa_guideline.md
```

권장 문서 포맷:

```text
# Source
Title:
Organization / Journal:
Year:
URL:

# Definition

# Positive criteria

# Negative criteria

# Equivocal findings / exclusions

# Terminology and synonyms
```

## 3. Embedding Index 생성

```bash
!python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/build_index.py \
  --guidance-dir /kaggle/working/rsna_knee_guidance_rag_embedding/guidance \
  --output-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index
```

기본 임베딩 모델:

```text
intfloat/multilingual-e5-small
```

## 4. Retrieval 확인

```bash
!python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/inspect_retrieval.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --index-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index \
  --row 0
```

## 5. Expert 58개 검증

```bash
!python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --index-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index \
  --output-dir /kaggle/working/rag_validation
```

결과:
- `predictions.csv`
- `metrics_by_target.csv`
- `errors.csv`

## 6. 나머지 Report pseudo-label 생성

58개에서 충분히 검증한 후 실행하세요.

```bash
!python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/label_unlabeled.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --index-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index \
  --output /kaggle/working/train_pseudo_labels.csv
```

## Retrieval 구조

각 target마다 해당 target의 guidance 문서만 후보로 제한한 뒤 dense retrieval을 수행합니다.

```text
Report
  ↓
multilingual-e5-small
  ↓
query embedding

Guidance chunks
  ↓
multilingual-e5-small
  ↓
document embeddings

cosine similarity
  ↓
target별 top-k guidance
  ↓
중복 제거 + context cap
  ↓
Mistral
  ↓
12 labels
```

E5 권장 형식에 맞춰:
- query: `query: ...`
- passage: `passage: ...`

prefix를 자동으로 붙입니다.
