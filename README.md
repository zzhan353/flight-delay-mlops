# Flight Delay MLOps

**[Live demo](https://ca-flightdelay.salmonbush-4ca37403.eastus.azurecontainerapps.io)** &middot;
predicts the probability that a US domestic flight arrives 15+ minutes late.

Trained on 2.2M flights from the US DOT Bureau of Transportation Statistics, joined
with NOAA weather. The interesting part is not the model — it is the machinery that
decides whether the model is allowed to ship, and that tells me when it stops working.

The whole stack runs for **under $1/month**.

---

## What this actually demonstrates

Most portfolio ML projects report a flattering accuracy number on a static dataset.
This one is built around the parts that decide whether a model survives contact with
production:

| Concern | How it is handled here |
|---|---|
| **Target leakage** | A feature contract in [`schema.py`](src/flight_delay/schema.py) names 25 post-departure columns that must never reach the model. Enforced on every data build and by unit tests, not by code review. |
| **Temporal validity** | Chronological train/holdout split. A random split would train on flights that happen *after* the ones being scored. |
| **Deployment safety** | A [gate](src/flight_delay/evaluate.py) blocks any model that fails to beat a lookup-table baseline, is badly calibrated, or regresses against what is currently serving. |
| **Honest evaluation** | Every run scores a deliberately dumb baseline on the same holdout. "PR-AUC 0.28" means nothing without "versus what?" |
| **Drift** | Monthly job scores the deployed model against ground truth. Alerts on *performance decay*, not on input drift — see below for why that distinction matters. |
| **Cost** | Architecture chosen under a hard budget. The reasoning is documented, because "it depends" is the real answer to most infrastructure questions. |

---

## Architecture

```
BTS monthly archive ──┐
                      ├──► data build ──► parquet ──► train ──► gate ──► ghcr.io image
NOAA daily weather ───┘      (CI)                     (CI)      (CI)          │
                                                                              ▼
                                            Azure Container Apps (scale-to-zero)
                                                                              │
                                            Azure ML workspace ◄── registry / lineage
```

**Training runs in GitHub Actions, not on Azure ML compute.** At this scale — 500k rows
per month, ~90 seconds to fit — a CI runner trains the model for free, while an Azure ML
cluster costs $0.29/hour plus a 10-minute environment image build on first use. Azure ML
is used for what it is uniquely good at: model registry, lineage and experiment
comparison. The training entry point still takes `--backend azureml`, so the same code
moves to a cluster unchanged when the data outgrows that argument.

**No Azure Container Registry.** ACR Basic is ~$5/month, more than the entire budget.
Images go to GitHub Container Registry, which is free for public repositories, and the
Container App pulls them without credentials — so there is no registry secret to store,
rotate or leak.

**Scale to zero.** `minReplicas: 0` means an idle service costs nothing. The trade is a
cold start of a few seconds on the first request after a quiet period; the model loads
in ~690ms, and it is baked into the image rather than fetched at boot precisely so that
path stays short.

Infrastructure is [Bicep](infra/main.bicep). Azure authentication from CI uses OIDC
federation scoped to this repository's `main` branch — there is no client secret
anywhere in the pipeline.

---

## Results

Trained on Jan–Mar 2025, evaluated on April 2025 (578k flights it never saw):

| Metric | Baseline | Model | |
|---|---|---|---|
| **Top-decile lift** | 1.552 | **1.721** | +11% |
| PR-AUC | 0.2696 | 0.2798 | +3.8% |
| ROC-AUC | 0.6207 | 0.6218 | +0.2% |
| Calibration error (ECE) | 0.0102 | 0.0121 | — |

The baseline is not a coin flip: it is a lookup table of historical delay rates by
carrier, origin airport and departure hour. Beating it is the bar that matters.

**An honest reading: machine learning adds a modest amount here.** Of the 10% of flights
the model flags as riskiest, 1.72× as many are actually late compared with the average
flight — a real improvement over the lookup table's 1.55×, and useful for triage. But
overall ranking is barely better. Most of what makes a flight late is same-day network
state: the inbound aircraft, crew rotations, ATC ground stops. A model that sees the
schedule and the season cannot see any of it.

That conclusion came out of this pipeline rather than around it, which is the point.

---

## Three decisions worth reading the code for

### 1. The gate caught the model being overconfident, so I fixed the model

The first trained model had an ECE of 0.047 and the gate blocked it. Inspecting the
calibration curve showed why: flights it rated at 74% were actually late 38% of the
time. Boosted trees rank well but push probabilities away from the base rate, and the
demo shows a percentage to a human being.

The fix is isotonic calibration fitted on the **trailing two weeks of the training
window** — not on the holdout, which would leak the evaluation set, and not on data the
model was fitted on, which would just re-learn the overconfidence. ECE fell to 0.0121.

### 2. When the model missed the bar, I changed what the gate measures — and said so

The gate originally required PR-AUC lift ≥ 1.05× over the baseline. The model reached
1.038×. The easy move is to lower the threshold to 1.03 and let it through, which is
exactly how gates become decorative.

Instead: which number does the product decision actually rest on? This model exists to
flag risky flights, so the operative metric is top-decile lift, where it reaches 1.72×
against 1.55× — an 11% improvement that overall PR-AUC dilutes across the other nine
deciles. PR-AUC is retained as a **floor** (the model must not rank *worse* than the
lookup table), and both numbers print on every run so a reviewer sees the full picture
rather than the flattering half.

Changing a threshold after seeing results deserves suspicion, so the reasoning lives in
[`evaluate.py`](src/flight_delay/evaluate.py) where it can be argued with.

### 3. Drift monitoring alerts on decay, not on drift

Running the drift check from January to April reports PSI above **3.0** on temperature —
enormous by any conventional reading — while live PR-AUC is unchanged to four decimal
places. The seasons changing is exactly the variation the model was trained across.

Wiring `PSI > 0.25` straight to a retrain trigger is the most common way drift
monitoring becomes an expensive noise generator. Here, performance decay against ground
truth is the decisive signal and opens a GitHub issue; input drift is recorded as
context, so that when decay does appear there is already an answer to "what changed?"

BTS publishes labels alongside flights, so — unusually for a production system — last
month's predictions can be scored against truth.

---

## Known limitations

Stated because a reviewer will find them anyway, and because they bound what the numbers
mean:

- **Weather is observed, not forecast.** Training joins the weather that actually
  occurred. At booking time only a forecast exists, so offline metrics are optimistic by
  roughly 1–2 points of PR-AUC. The serving API accepts weather but the demo sends none,
  which means live predictions run without it entirely.
- **Weather covers 64% of flights.** 30 airports are mapped to NOAA stations; the rest
  get nulls, which the model handles natively.
- **No same-day network state.** The single biggest driver of delay — a late inbound
  aircraft — is absent, and it is why ROC-AUC sits at 0.62 rather than 0.75+.
- **The model artifact is cloudpickle, not skops.** A custom transformer forces it. The
  boundary is stated in [`train.py`](src/flight_delay/train.py): artifacts are produced
  only by this repo's CI and never loaded from user-supplied paths. On a shared artifact
  store, this decision would need revisiting.
- **Four months of data.** Enough for a chronological split and a real drift check, not
  enough to model annual seasonality.

---

## Running it

```bash
make install          # uv venv + dependencies
make data             # download BTS + NOAA, build parquet (~15s/month)
make train            # train, evaluate against baseline, write metrics
make serve            # http://localhost:8000
make test             # 37 tests
```

Deploy the infrastructure:

```bash
az deployment group create -g rg-flight-delay --template-file infra/main.bicep
```

## Layout

```
src/flight_delay/
  schema.py             feature contract and the leakage denylist
  data/bts.py           BTS download and cleaning
  data/weather.py       NOAA join, 30 mapped airport stations
  features.py           pipeline: encoders, model, chronological split, calibration
  schedule_context.py   congestion and schedule-padding features
  baseline.py           the lookup table the model must beat
  metrics.py            ranking, calibration and business metrics
  train.py              training entry point, local or Azure ML
  evaluate.py           the deployment gate
  monitor.py            drift and decay
  serve.py              FastAPI service and demo page
infra/main.bicep        all Azure resources
.github/workflows/      CI, deploy, monthly monitor
```

---

Data: [BTS On-Time Performance](https://transtats.bts.gov/), [NOAA GHCN-Daily](https://www.ncei.noaa.gov/).
Built by [zzhan353](https://github.com/zzhan353) &middot; zzhan353@asu.edu
