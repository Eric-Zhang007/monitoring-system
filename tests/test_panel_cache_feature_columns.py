from __future__ import annotations

from training.cache import panel_cache as cache_mod


class _FakeCursor:
    def __init__(self):
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):  # noqa: ANN001
        _ = sql
        table = str((params or [""])[0])
        if table == "feature_matrix_main":
            self._rows = [
                {"column_name": "as_of_ts"},
                {"column_name": "feature_values"},
                {"column_name": "feature_mask"},
                {"column_name": "schema_hash"},
            ]
        else:
            self._rows = []

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_feature_matrix_vector_columns_support_legacy_names():
    layout = cache_mod._feature_matrix_layout(_FakeConn())
    assert layout["mode"] == "vector"
    assert layout["values_col"] == "feature_values"
    assert layout["mask_col"] == "feature_mask"
