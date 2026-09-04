# Multivariate Time Series Forecasting with a Temporal Convolutional Network

Group 10 — Deep Learning bonus project, SS26, TU Darmstadt.
Members: Jessica Hassibi, Sardorbek Bobomurodov, Yulia Belyaeva, Felix Bauer

We forecast 336 hours ahead for each series of the course benchmark
(`operations_forecasting_2026`) with a dilated **temporal convolutional network** trained in
PyTorch, and we repeat the experiment on a second, independently chosen dataset
(**ASHRAE GEPIII** building electricity meters) to test whether the architecture transfers.

---

## 1. Reproducing everything
- **run commands from the repository root**
- The dataset and training scripts are the exception: their default paths are relative to `submission/`, so
those blocks start with `cd submission`.

### 1.1 Python environment

Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-dev.txt
```

- Activate with `source .venv/bin/activate`
- In PyCharm: Settings → Project → Python Interpreter → Add → Existing → `.venv/bin/python`

### 1.2 Benchmark dataset

Auto-downloads from Hugging Face (`AIML-TUDA/dlam-ts-project-data-2026`) into
`submission/dataset/` the first time any script needs it:

```bash
cd submission && ../.venv/bin/python check_dataset.py
```

Produces `train.csv`, `validation_input.csv`, `forecast_index_validation.csv`, `metadata.json`.

### 1.3 ASHRAE GEPIII dataset (Kaggle, needs auth)
1. **Authenticate**:
   ```bash
   .venv/bin/kaggle auth login
   ```
2. **Accept the competition rules** — required, or downloads return HTTP 403:
   visit https://www.kaggle.com/competitions/ashrae-energy-prediction/rules,
   click "Late Submission" if offered, then "I Understand and Accept".

- Download the three files.

```bash
mkdir -p submission/dataset_ashrae/raw
for f in building_metadata.csv weather_train.csv train.csv; do
  .venv/bin/kaggle competitions download -c ashrae-energy-prediction -f $f -p submission/dataset_ashrae/raw
done
cd submission/dataset_ashrae/raw && for z in *.zip; do unzip -o "$z" && rm "$z"; done
```

- Convert them into the same format as the benchmark, so one pipeline for both:

```bash
cd submission && ../.venv/bin/python convert_ashrae.py
```

Produces `train.csv`, `validation_input.csv`, `forecast_index_validation.csv`, `validation_target.csv`,`metadata.json` and a generated dataset card.

### 1.4 Training

Both runs use seed 42 and write TensorBoard logs and checkpoints into their log directory
(`tensorboard --logdir submission/logs`).

Benchmark:

```bash
cd submission && ../.venv/bin/python run_training.py tcn \
  --train_config.num_epochs 1 --train_config.dataset_stride 1 --train_config.seed 42
```

ASHRAE:

```bash
cd submission && ../.venv/bin/python run_training.py tcn \
  --dataset_dir dataset_ashrae --log_dir_name logs_ashrae \
  --train_config.num_epochs 1 --train_config.dataset_stride 7 --train_config.seed 42
```

Swap `tcn` for `linear` to train the linear reference model, or use
`run_training.py checkpoint --checkpoint <path> --log_dir_name <dir>` to continue a run.

### 1.5 Evaluation

- **`evalharness.py`** holds out the tail of the *training* series.

```bash
cd submission && ../.venv/bin/python evalharness.py \
  --checkpoint logs/tcn/<run>/checkpoint-<n>.pt --dataset-dir dataset
```

- **`score_predictions.py`** scores against genuinely held-out labels — ASHRAE only:

```bash
cd submission
../.venv/bin/python predict.py --checkpoint logs_ashrae/tcn/<run>/checkpoint-<n>.pt \
  --input_dir dataset_ashrae --output_file predictions/ashrae_tcn.csv
../.venv/bin/python score_predictions.py --predictions predictions/ashrae_tcn.csv \
  --labels dataset_ashrae/validation_target.csv
```

- Simple baselines (naive last value, lag-24, lag-168, seasonal mean):

```bash
.venv/bin/python baseline/run_baselines.py \
  --train submission/dataset/train.csv \
  --forecast-index submission/dataset/forecast_index_validation.csv \
  --output-dir submission/dataset/baselines
```

  Swap `dataset` for `dataset_ashrae` in all three paths to get the ASHRAE baselines, which
  are the ones `score_predictions.py` above can actually score against real labels.

### 1.6 Inference and the final archive

- The command required for private evaluation:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

---

## 2. Notes
- All hyperparameters are recorded per run in `config.yml` next to each checkpoint.
