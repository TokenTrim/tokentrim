from __future__ import annotations

import pytest

from tokentrim.transforms.base import Transform


def test_tool_step_is_abstract() -> None:
    with pytest.raises(TypeError):
        Transform()
