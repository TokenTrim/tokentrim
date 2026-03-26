from __future__ import annotations

import pytest

from tokentrim.integrations.base import IntegrationAdapter


def test_integration_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        IntegrationAdapter()
