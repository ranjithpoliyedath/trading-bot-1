"""
bot/models — trading model registry and implementations.

To use a model:
    from bot.models.registry import get_model, list_models

    # See all available models (built-in + custom)
    for meta in list_models():
        print(meta.id, '-', meta.name)

    # Get an instance and predict
    model = get_model("rsi_macd_v1")
    signal, conf = model.predict(feature_row)
"""

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model, get_model, list_models

__all__ = [
    "BaseModel", "ModelMetadata", "Signal",
    "register_model", "get_model", "list_models",
]
