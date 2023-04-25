from river.datasets import base
from river.ensemble import AdaptiveRandomForestClassifier, AdaptiveRandomForestRegressor
from ixai.utils.wrappers import RiverWrapper


def _get_arf_model(task, **kwargs):
    if task == base.BINARY_CLF:
        model = AdaptiveRandomForestClassifier(**kwargs)
    else:
        model = AdaptiveRandomForestRegressor(**kwargs)
    return model


def get_model(model_name, task, feature_names, **model_kw):
    if model_name == 'ARF':
        model = _get_arf_model(task, **model_kw)
    else:
        raise NotImplementedError
    model_function = RiverWrapper(model.predict_one)
    return model, model_function
