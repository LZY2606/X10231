"""ADCP beam velocity -> instrument / ship / earth velocity evidence core.

All domain rules in this package are pure functions so they can be replayed
deterministically. Every rule module exposes a ``*_RULE_VERSION`` constant;
those versions participate in the derivation fingerprint (see
``adcp.fingerprint``).
"""

from .units import Raw, normalize, ConversionResult

__all__ = ["Raw", "normalize", "ConversionResult"]
