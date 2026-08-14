# Kaggle RSNA Knee Abnormality Detection — Guidance-based Report Labeling

Radiology reports are classified into the 12 competition targets using a
target-specific medical guidance document and an open-source instruction LLM.

This version intentionally does **not** use RAG, embeddings, chunking, or vector search.
There are only 12 fixed targets, and each guidance file is compact enough to provide
directly to the LLM.

## Architecture

```text
Report + acl.md              -> Mistral -> ACL
Report + mcl.md              -> Mistral -> MCL
Report + medial_meniscus.md  -> Mistral -> Medial Meniscus
...
Report + fracture.md         -> Mistral -> Fracture
```

Each report is therefore evaluated independently for all 12 targets.

## Why direct guidance injection?

The task has a fixed and small knowledge set:
- 12 targets
- one compact literature-derived guidance document per target

Adding embeddings/retrieval would introduce an extra failure point:
the clinically relevant rule can be omitted by retrieval even though the complete
guidance easily fits into the LLM context.

The 58 expert-labeled reports are used only for evaluation. The guidance files were
summarized from external medical/radiology literature rather than reverse-engineered
from those labels.

## Structure

```text
.
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
├── scripts/
│   ├── validate_expert.py
│   └── label_unlabeled.py
├── src/knee_guidance/
│   ├── __init__.py
│   ├── classifier.py
│   ├── constants.py
│   ├── evaluation.py
│   ├── guidance.py
│   ├── llm.py
│   └── prompting.py
├── kaggle_quickstart.py
├── pyproject.toml
└── requirements-kaggle.txt
```

## Model

Default:
`mistralai/Mistral-7B-Instruct-v0.3`

Default loading:
4-bit NF4 via `bitsandbytes`.

## Kaggle setup

```bash
!pip install -q bitsandbytes
!pip install -e /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean --no-deps
```

## Quick smoke test

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/guidance \
  --output-dir /kaggle/working/guidance_validation_test \
  --limit 2
```

## Full expert-label validation

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/scripts/validate_expert.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/guidance \
  --output-dir /kaggle/working/guidance_validation
```

Outputs:
- `predictions.csv`
- `metrics_by_target.csv`
- `errors.csv`
- `raw_outputs.csv`

## Pseudo-label unlabeled reports

Run only after validating the approach on the expert-labeled subset.

```bash
!python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/scripts/label_unlabeled.py \
  --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
  --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/guidance \
  --output /kaggle/working/train_pseudo_labels.csv
```

## Removed from the previous embedding-RAG version

The following components are deliberately removed:
- embedding model
- `embeddings.py`
- chunking
- vector index generation
- `indexer.py`
- retriever
- retrieval inspection
- `build_index.py`
- RAG pipeline/context aggregation

They are unnecessary for the current fixed 12-document design.
