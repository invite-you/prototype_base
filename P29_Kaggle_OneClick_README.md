# P29 Kaggle One-Click

Import `P29_Kaggle_OneClick_DualHead.ipynb` into Kaggle.

Settings:
- Internet: ON
- Accelerator: GPU

Then choose **Run All**. No file uploads or patch cells are required.

The notebook:
- directly downloads the pinned DeBERTa snapshot with requests;
- loads tokenizer.json directly and then runs Transformers offline;
- builds the NLI -> Binary-RE intermediate encoder;
- runs exactly three P29 conditions;
- uses the same fair checkpoint selection rule as P20/P27;
- never opens SEALED_HOLDOUT_V2;
- writes only compact CSV/JSON files to `/kaggle/working/pii_p29_results.zip`.

Expected PASS markers:
- DIRECT TOKENIZER PREFLIGHT: PASS
- LOCAL MODEL FORWARD PREFLIGHT: PASS
- P29 AUXILIARY SUPERVISION INTEGRITY: PASS
- P29 FULL PREFLIGHT: PASS
- P29 EXECUTION INTEGRITY: PASS
- P29 RESULT ZIP INTEGRITY: PASS
