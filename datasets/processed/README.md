# Processed data

This folder holds the **cleaned dataset** used for modeling.

## `processed_day.csv`

It is the raw `datasets/raw/day.csv` after removing the columns we don't use:

- `instant` — row id
- `dteday` — date
- `casual` + `registered` — they sum exactly to `cnt`, so keeping them would be data leakage

What's left is the **11 features + the target `cnt`** (731 rows).

## Important

This file is a **record / output**, not an input. Nothing reads it back:

- Training (`src/train.py`) always reads the raw `datasets/raw/day.csv`, cleans it
  in memory, and writes this file as a snapshot (via `save_processed_data`).
- The actual feature transformations (one-hot encoding, polynomial terms) happen
  **inside the model pipeline** (`models/best_model.pkl`), not here.

So this CSV exists just to make the "clean data we model on" visible and traceable.
It is regenerated automatically every time the model is trained.
