from __future__ import annotations

import pytest

from tokentrim.context.base import ContextStep


def test_context_step_is_abstract() -> None:
    with pytest.raises(TypeError):
        ContextStep()
