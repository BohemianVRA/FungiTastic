"""Typing objects for the Regex library."""
from typing import AnyStr
import regex  # type: ignore[import]
import sys

from typing import _alias  # type: ignore[attr-defined]

if sys.version_info >= (3, 9):
    Pattern = _alias(type(regex.compile('')), 1)
    Match = _alias(type(regex.compile('').match('')), 1)
else:
    Pattern = _alias(type(regex.compile('')), AnyStr)
    Match = _alias(type(regex.compile('').match('')), AnyStr)
