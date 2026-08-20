"""Load a trained model without depending on MLflow.

Training writes the model with `mlflow.sklearn.save_model`, which produces a directory
containing `model.pkl` — a cloudpickled scikit-learn Pipeline — plus metadata. Reading
it back through `mlflow.sklearn.load_model` would be the obvious choice, but it pulls
MLflow into the serving image, and MLflow arrives with matplotlib, PIL, SQLAlchemy and
a dependency on pyarrow: roughly 110 MB of packages that are never touched while
answering a request.

Since the pickle only references classes that live in this package, the standard
library's `pickle` can load it directly. That keeps the training side idiomatic — the
artifact is still a real MLflow model, registered and versioned as one — while the
serving image carries only what inference needs.

The trade is losing MLflow's schema enforcement at load time. That is acceptable here
because the API validates every request through Pydantic before a frame is built,
which is a stricter check than the model signature and produces better error messages.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

MODEL_FILENAME = "model.pkl"


def load_model(model_dir: Path | str) -> Any:
    """Load the pickled estimator from an MLflow model directory."""
    path = Path(model_dir) / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {MODEL_FILENAME} in {model_dir}. Expected an MLflow model directory "
            "produced by `python -m flight_delay.train`."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)
