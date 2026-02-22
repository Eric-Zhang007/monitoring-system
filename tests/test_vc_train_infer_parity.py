from __future__ import annotations

import numpy as np

from vc.feature_spec import vector_from_context, vector_from_event_row


def test_vc_train_infer_feature_parity():
    row = {
        "event_type": "funding",
        "source_tier": 2,
        "confidence_score": 0.85,
        "event_importance": 0.7,
        "novelty_score": 0.5,
    }
    v_train = vector_from_event_row(row)
    v_infer = vector_from_context([row])
    assert np.array_equal(v_train, v_infer)
