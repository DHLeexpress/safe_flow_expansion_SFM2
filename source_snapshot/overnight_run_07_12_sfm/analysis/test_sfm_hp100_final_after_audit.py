import pytest

import sfm_hp100_final_after_audit as A


def test_replicas_are_unique_and_order_preserving():
    assert A._parse_replicas("2,12,4") == (2, 12, 4)
    with pytest.raises(ValueError):
        A._parse_replicas("2,2")
