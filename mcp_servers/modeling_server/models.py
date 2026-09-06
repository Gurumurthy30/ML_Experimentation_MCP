MODEL_REGISTRY = {
    "linear": {"regression": Ridge, "classification": LogisticRegression},
    "tree_ensemble": {"regression": HistGradientBoostingRegressor,
                       "classification": HistGradientBoostingClassifier},
}
"""Deliberately small and fixed -- model choice is not a search dimension
in this project. hyperparams passed to cross_validate_model overrides
these defaults, never replaces the family."""

def get_model(model_family: str, task_type: str, hyperparams: dict = None):
    """Looks up MODEL_REGISTRY[model_family][task_type], instantiates it
    with hyperparams merged over its defaults."""

def compute_model_id(folds_id: str, pipeline_spec_id: str, model_family: str, hyperparams: dict) -> str:
    """sha1 of the sorted, JSON-serialized tuple of all four inputs.
    Deterministic -- finalize_model checks this before fitting, so calling
    it twice with identical inputs returns the existing model_id instead
    of refitting."""