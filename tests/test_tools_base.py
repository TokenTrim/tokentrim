from __future__ import annotations

import pytest

from tokentrim.tools.base import ToolStep


def test_tool_step_is_abstract() -> None:
    with pytest.raises(TypeError):
        ToolStep()
