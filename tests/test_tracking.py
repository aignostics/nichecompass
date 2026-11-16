import sys

import pytest

from nichecompass.tracking import import_mlflow


def test_import_mlflow_explains_the_missing_extra(monkeypatch):
    # A None entry in sys.modules makes the import fail like a missing package.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match=r"nichecompass\[tracking\]"):
        import_mlflow()
