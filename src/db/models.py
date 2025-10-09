"""Model registry and path utilities.

This module provides backwards compatibility for legacy code.

DEPRECATED: This module is deprecated. Use src.utils.model_paths instead.

All functionality has been moved to:
- src.utils.model_paths.get_model_xml_path()
- src.utils.model_paths.find_model_xml()
- src.db.registry.MODEL_REGISTRY
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "src.db.models is deprecated and will be removed in a future version. "
    "Use src.utils.model_paths and src.db.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backwards compatibility
from src.db.registry import MODEL_REGISTRY  # noqa: E402, F401
from src.utils.model_paths import get_model_xml_path  # noqa: E402, F401
