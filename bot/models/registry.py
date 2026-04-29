"""
bot/models/registry.py
-----------------------
Central registry where every model registers itself.

Models register at import time using the @register_model decorator.
The dashboard queries the registry to populate dropdowns.

Custom user-built models loaded from JSON files in dashboard/custom_models/
are also surfaced through this registry.
"""

import json
import logging
import re
from pathlib import Path
from typing import Type

from bot.models.base import BaseModel, ModelMetadata

# Same allowlist the save endpoints enforce — defends _load_custom_model
# against path traversal if a model_id arrives from an unvalidated source.
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,50}$")

logger = logging.getLogger(__name__)

CUSTOM_MODELS_DIR = Path(__file__).parent.parent.parent / "dashboard" / "custom_models"

_registry: dict[str, Type[BaseModel]] = {}


def register_model(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    Decorator to register a model class.

    Usage:
        @register_model
        class MyModel(BaseModel):
            metadata = ModelMetadata(...)
    """
    if not hasattr(model_cls, "metadata"):
        raise ValueError(f"{model_cls.__name__} must define a metadata attribute.")

    model_id = model_cls.metadata.id
    if model_id in _registry:
        logger.warning("Model id %r already registered — overwriting.", model_id)

    _registry[model_id] = model_cls
    logger.debug("Registered model: %s (%s)", model_id, model_cls.__name__)
    return model_cls


def get_model(model_id: str) -> BaseModel:
    """
    Instantiate and return a model by id.

    Args:
        model_id: e.g. 'rsi_macd_v1' or 'custom:my_strategy'

    Returns:
        BaseModel instance ready for predict() calls.
    """
    if model_id.startswith("custom:"):
        return _load_custom_model(model_id[len("custom:"):])

    _ensure_builtin_imports()
    if model_id not in _registry:
        raise KeyError(f"Unknown model id: {model_id!r}. Available: {list(_registry)}")

    return _registry[model_id]()


def list_models() -> list[ModelMetadata]:
    """Return metadata for every registered model (built-in + custom)."""
    _ensure_builtin_imports()

    builtin = [cls.metadata for cls in _registry.values()]
    custom  = _list_custom_models()
    return builtin + custom


def _ensure_builtin_imports():
    """Import built-in modules so their @register_model decorators run."""
    try:
        from bot.models.builtin import (   # noqa: F401
            rsi_macd_v1, bollinger_v1, sentiment_v1,
            qullamaggie_v1, vcp_v1,
            golden_cross_v1, donchian_v1, connors_rsi2_v1, ibs_v1,
            adx_trend_v1, keltner_breakout_v1, obv_momentum_v1,
            zscore_reversion_v1,
            jt_momentum_v1,
            # ── 2026-04-29: research + Reddit additions ──
            tsmom_v1, pct52w_high_v1, recovery_rally_v1,
            weinstein_v1, sector_rotation_v1,
        )
    except ImportError as exc:
        logger.warning("Could not import built-in models: %s", exc)


def _list_custom_models() -> list[ModelMetadata]:
    if not CUSTOM_MODELS_DIR.exists():
        return []

    metadata = []
    for path in CUSTOM_MODELS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            metadata.append(ModelMetadata(
                id          = f"custom:{data['id']}",
                name        = data.get("name", data["id"]),
                description = data.get("description", ""),
                type        = "custom",
                required_features = data.get("required_features", []),
            ))
        except Exception as exc:
            logger.warning("Skipping invalid custom model %s: %s", path, exc)
    return metadata


def _load_custom_model(custom_id: str) -> BaseModel:
    """Load a custom model from its JSON file and wrap it as a BaseModel.

    Defensive: validate ``custom_id`` against the same allowlist the
    save endpoints use, and verify the resolved path stays inside
    ``CUSTOM_MODELS_DIR`` so a crafted ``custom:../..`` model_id can't
    read JSON from outside the intended directory.
    """
    from bot.models.custom import CustomRuleModel    # local import to avoid cycle

    if not _CUSTOM_ID_RE.match(custom_id or ""):
        raise KeyError(f"Invalid custom model id: {custom_id!r}")

    path = (CUSTOM_MODELS_DIR / f"{custom_id}.json").resolve()
    try:
        path.relative_to(CUSTOM_MODELS_DIR.resolve())
    except ValueError:
        raise KeyError(f"Refusing to load custom model outside the dir: {custom_id!r}")

    if not path.exists():
        raise KeyError(f"Custom model not found: {custom_id}")

    with open(path) as f:
        spec = json.load(f)
    return CustomRuleModel(spec)
