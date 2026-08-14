# %% [markdown]
# Cell 1 - install
# !pip install -q bitsandbytes
# !pip install -e /kaggle/working/rsna_knee_guidance_rag_embedding --no-deps

# %% [markdown]
# Cell 2 - build dense embedding index
# !python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/build_index.py \
#   --guidance-dir /kaggle/working/rsna_knee_guidance_rag_embedding/guidance \
#   --output-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index

# %% [markdown]
# Cell 3 - inspect retrieval
# !python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/inspect_retrieval.py \
#   --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
#   --index-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index \
#   --row 0

# %% [markdown]
# Cell 4 - validate 58 expert rows
# !python /kaggle/working/rsna_knee_guidance_rag_embedding/scripts/validate_expert.py \
#   --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
#   --index-dir /kaggle/working/rsna_knee_guidance_rag_embedding/artifacts/index \
#   --output-dir /kaggle/working/rag_validation

# %% [markdown]
# Cell 5 - inspect results
# import pandas as pd
# display(pd.read_csv("/kaggle/working/rag_validation/metrics_by_target.csv"))
# display(pd.read_csv("/kaggle/working/rag_validation/errors.csv").head(20))
