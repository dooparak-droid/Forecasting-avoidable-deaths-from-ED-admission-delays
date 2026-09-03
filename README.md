# Forecasting Avoidable Deaths from ED Admission Delays

An entry to the [SPHERE-PPL NHS forecasting competition](https://github.com/SPHERE-PPL/NHS-EAD-forecast), built with Michael Bakalov. The task was to forecast daily avoidable deaths from emergency department admission delays in the Bristol NHS system (BNSSG ICS), over rolling 10-day windows, using near-real-time operational data. The submitted model, a direct multi-horizon Random Forest, beat every baseline tested in cross-validation, and scored an MSE of 0.098 (days 1–5) and 0.110 (days 6–10) on the real assessment period.

The competition's dataset, rules, and template are located at the link above. This repo has the scripts and the final submission only. The \~16.7M-row raw dataset isn't included.

## Data

Raw data is \~16.7M rows in long format, loaded with `data.table::fread()`. Sub-daily records are aggregated to daily resolution using a metric-specific aggregation map (mean, max, sum, last, or as-is, depending on the metric). Metrics flagged `YESTERDAY` in their name are shifted back one day to align with the observation date. Negative values (`-9999` dummy codes) are treated as missing. Columns missing more than 50% of the time over the development period are dropped, using development-period rows only, so assessment-period patterns can't affect which columns are kept.

Origin date 2025-10-03 is absent from the assessment dataset, so the forecast period starting there (predicting 4–13 October) produces NA predictions, excluded from MSE via `na.rm = TRUE`.

## Feature engineering

Feature engineering is a single deterministic function, `build_features()` in `scripts/02_feature_eng.R`. Imputation is deliberately left out of this step and applied separately inside each cross-validation fold, using training-period medians only, so no validation-period information leaks into it.

The model uses three groups of features. The first is outcome lags, at 3 to 14 days, plus 3- and 7-day rolling means and a momentum term (lag-3 minus lag-6). Only lags of 3 days or more are used, since the outcome itself has a 3-day reporting lag, and anything shorter wouldn't be observable at prediction time. The second is calendar features: day of week (numeric and ordered factor), sine/cosine encodings of day-of-year for smooth seasonality, ISO week, month, a linear time index, and four binary flags for periods where NHS demand shifts structurally (New Year, Christmas week, Easter, summer). The third is ten clinically curated predictors spanning four domains: community capacity, acute-to-community transfer backlog, acute hospital pressure, and emergency demand. Each of these ten has 1-, 2-, and 3-day lags, plus 3- and 7-day rolling means.

Predictors were chosen for clinical distinctiveness, not raw correlation rank, to avoid redundant correlated pairs. Features were selected using the maximum absolute Spearman correlation against the outcome across horizons 3–10, computed on pre-2025 data only. Four composite features collapse correlated same-domain predictors into single signals, and every lagged or derived predictor has a missingness indicator, since a system failing to report is informative. The final feature matrix has 133 numeric columns.

## Cross-validation

An expanding-window design: 8 folds, fixed 10-day validation windows, and a 10-day gap between training end and validation start to match the forecast horizon, so no validation-period information reaches training. Median imputation is computed from training rows within each fold only, then applied to both the training and validation slices, mirroring how development-period medians are later applied to the assessment data.

## Models

Three baselines: a naive lag-7 forecast (same weekday, previous week), a 3-day rolling mean, and a linear regression per horizon using calendar features and outcome lags. The two main models are Random Forest (`ranger`) and XGBoost. Both use direct multi-horizon forecasting rather than a recursive one-step-ahead approach. A separate model is trained per horizon from 1 to 10 days out, so errors from one horizon don't compound into the next.

| Model             | MSE days 1–5 | MSE days 6–10 |
|-------------------|--------------|---------------|
| Random Forest     | **0.0834**   | **0.0790**    |
| XGBoost           | 0.0927       | 0.0860        |
| Linear regression | 0.114        | 0.109         |
| Rolling mean      | 0.111        | 0.180         |
| Naive (lag-7)     | 0.153        | 0.140         |

Random Forest was selected for both horizon groups and refit on all 927 development rows for submission.

## Residual analysis and bias correction

Out-of-fold residuals were reconstructed by refitting on each fold's training slice and collecting validation predictions across all 8 folds. Two systematic patterns are visible in them. One is a weekly cycle, with Saturday predictions consistently low. The other is a seasonal one, with January underpredicted, coinciding with the post-holiday surge. Two additive corrections were computed from these residuals. A day-of-week correction was applied per target date, and a monthly correction was applied uniformly across all 10 horizons for a given origin date.

On cross-validation data, MSE was lower with the corrections applied. On the real assessment period, it was higher instead:

| Version                        | MSE days 1–5 | MSE days 6–10 |
|--------------------------------|--------------|---------------|
| Uncorrected (primary)          | **0.0980**   | **0.110**     |
| Bias-corrected (supplementary) | 0.103        | 0.127         |

The monthly correction was estimated mostly from non-winter development data, and the assessment window is winter-dominated, so the correction was too large for that period. Uncorrected predictions were submitted as primary (`submission/pred_matrix.csv`). The bias-corrected version is kept as supplementary (`submission/pred_matrix_corrected.csv`).

## Limitations

Fold 5 (the January 2025 post-holiday surge) dominates cross-validation variance and is largely irreducible with this data. There aren't enough surge examples to train against. Holiday binary flags added negligible improvement for the same reason.

The monthly bias correction for January and February is estimated from very few cross-validation observations (fold 5 only), which is part of why it failed to generalise to the real assessment period.

Random Forest averages across many trees, which pulls predictions toward the centre of the distribution. It tends to underpredict on high-mortality days and overpredict slightly on low-mortality days, visible in the actual-vs-predicted scatter at horizon 1.

## Running it

Scripts run in order from the project root: `01_eda.R`, `02_feature_eng.R`, `03_modeling.R`, `04_residual_analysis.R`, `05_forecast.R`. Running them needs the competition's development and assessment CSVs under `data/`, not included here but available from the [SPHERE-PPL repo](https://github.com/SPHERE-PPL/NHS-EAD-forecast) linked above. `submission/SPHERE_PPL_report.Rmd` renders the full competition report.

## License

MIT, see [LICENSE](LICENSE), for the code in this repo. The forecasting task, dataset, and competition rules belong to SPHERE-PPL.
