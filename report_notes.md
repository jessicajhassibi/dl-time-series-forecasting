# Development Notes for the Report

Multivariate time-series forecasting with deep learning — TCN approach.
These notes document what was implemented and the reasoning behind each decision,
organised so the content can be lifted into the report's Method and Experiments sections.

## 1. Problem Setup

The task is to forecast a future hourly operational load index for each `series_id`.
The provided dataset (`AIML-TUDA/dlam-ts-project-data-2026`) contains a training split,
a validation input split, and forecast indices. The validation and private test
forecast indices each contain 336 hourly timestamps per series. Models are scored by
MAE, MSE, RMSE, MAPE, sMAPE, and WAPE (lower is better on all). The explicit goal is to
beat the provided `seasonal_mean` baseline.

Each training sample provides a context window of past target values plus 22 auxiliary
feature channels (calendar/exogenous features), and the corresponding future target
values to predict.

## 2. Reproducible Environment

To ensure reproducibility (a stated grading requirement), the project was set up in an
isolated virtual environment:

- Python 3.13 with a dedicated `venv`.
- Dependencies pinned via `submission/requirements.txt` (PyTorch >= 2.2, pandas, numpy, etc.).
- The dataset downloads automatically from the Hugging Face Hub on first use and is cached
  locally under `submission/dataset/`.
- Random seeds are fixed through the existing `TrainConfig.set_seed` (NumPy and PyTorch),
  default seed 42.

The pipeline was verified end-to-end before modelling: `check_dataset.py` confirmed the
data loads and produces the expected tensor shapes, and a one-epoch run of the provided
`linear_features` model confirmed the training loop, validation, and checkpointing all work.

## 3. Baselines and Reference Point

The repository ships three reference models:

- `linear`: a single linear layer with reversible instance normalisation (RevIN),
  operating on the target channel only.
- `linear_features`: the linear model extended with a second linear layer that mixes the
  22 feature channels. A one-epoch training run reached a validation WAPE of **0.303**
  (on the internal 90/10 split). This is our internal reference to beat.
- `tcn`: a placeholder stub (`return x + bias`) marked `# TODO implement tcn`. This is the
  slot intended for a real deep model and the starting point for our contribution.

Simple statistical baselines (`naive_last_value`, `lag24_repeat`, `lag168_repeat`,
`seasonal_mean`) were generated from `baseline/run_baselines.py`. Per the assignment,
`seasonal_mean` is the official baseline our model must improve on.

## 4. Method: Temporal Convolutional Network (TCN)

We implement a TCN following Bai, Kolter & Koltun (2018), "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling". A TCN is well suited
to this problem: it captures long-range temporal dependencies through dilated causal
convolutions, trains faster and more stably than recurrent models, and has a receptive
field that can be sized to cover the full context window.

### 4.1 Residual block (matches Bai et al. Figure 1b)

Each temporal block contains two **dilated causal convolutions**, each followed by
**weight normalisation**, a **ReLU** activation, and **dropout**, wrapped in a
**residual connection** (with a 1x1 convolution on the skip path when input and output
channel counts differ). Causality is enforced by left-padding each convolution by
`(kernel_size - 1) * dilation` and then trimming the equal amount from the right end of
the output (the `Chomp1d` operation), so no output position depends on future inputs.

### 4.2 Dilation schedule and receptive field

Dilation grows exponentially across blocks: d = 1, 2, 4, 8, ... (`2**i` at block `i`).
With kernel size 3 and 8 blocks, the receptive field is approximately 1021 time steps,
which fully covers the default context window of 1008 hours (336 x 3). Every forecast
therefore has access to the entire provided history.

### 4.3 Input construction

The target channel and the 22 feature channels are concatenated into a single
`[batch, channels, length]` tensor (23 input channels), so the convolutions jointly model
the target and its exogenous features rather than the target in isolation.

### 4.4 Reversible Instance Normalisation (RevIN)

Before the convolutional stack, the target channel is normalised per-instance using its
own mean and standard deviation, with learnable affine parameters (gamma, beta); the
inverse transform is applied to the model output. This mirrors the technique used by the
provided linear baselines and addresses the large scale differences between series, which
otherwise dominate the loss.

### 4.5 Single-shot forecasting head

The paper's original formulation produces a same-length output sequence for next-step
prediction. Because this benchmark requires a fixed 336-step horizon, we instead take the
representation at the final time step of the convolutional stack and project it through a
linear head to all `prediction_horizon` future values at once. This "single-shot" design
avoids autoregressive rollout and the associated error accumulation, and it fits the
existing inference code path: with `prediction_horizon = 336`, the forecast for each series
is produced in a single forward pass, so no feature-window advancement is required during
rollout.

### 4.6 Integration choices

The model is registered as a non-shifted output model (it returns only the future values,
not the shifted input). To keep the provided template files untouched, the model lives in a
new file (`src/models/tcn_deep.py`) and is trained through a dedicated standalone script
(`run_tcn.py`) that reuses the existing dataset loader, training loop, and checkpointing.
Model hyperparameters are written to a config file alongside each checkpoint so runs are
fully reconstructible.

## 5. Training Configuration

- Optimiser: AdamW (lr 1e-3, weight decay 1e-2) — from the provided training loop.
- Loss: mean squared error on the target.
- Batch size: 128. Validation every 500 steps and at each epoch end on a held-out 10% split.
- Metric tracked during training: WAPE.
- Default hyperparameters: context 1008, horizon 336, hidden 64, 8 levels, kernel 3,
  dropout 0.1. These are exposed as command-line flags for ablation.
- Logs and checkpoints written under `logs/tcn_deep/` and viewable in TensorBoard.

## 6. Experiments Planned

- Train the TCN to convergence and compare validation WAPE against `linear_features` (0.303)
  and, via the leaderboard, against `seasonal_mean`.
- Ablations over model capacity (hidden size, number of levels), regularisation (dropout),
  and context length.
- Report the full metric suite (MAE, MSE, RMSE, MAPE, sMAPE, WAPE) from the public
  validation leaderboard for the chosen checkpoint.

## 7. Reference

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.
