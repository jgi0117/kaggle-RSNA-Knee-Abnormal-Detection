# Cell 1
# !pip install -q bitsandbytes
# !pip install -e /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean --no-deps

# Cell 2 - quick test on 2 expert reports
# !python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/scripts/validate_expert.py \
#   --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
#   --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/guidance \
#   --output-dir /kaggle/working/guidance_validation_test \
#   --limit 2

# Cell 3 - full 58-study validation
# !python /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/scripts/validate_expert.py \
#   --csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
#   --guidance-dir /kaggle/working/kaggle-RSNA-Knee-Abnormal-Detection-clean/guidance \
#   --output-dir /kaggle/working/guidance_validation

# Cell 4 - inspect metrics
# import pandas as pd
# display(pd.read_csv("/kaggle/working/guidance_validation/metrics_by_target.csv"))
# display(pd.read_csv("/kaggle/working/guidance_validation/errors.csv").head(30))
