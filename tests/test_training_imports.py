"""The training stack imports (MLflow, the dataset classes, the loss factory, the trainer). It does not train."""
import importlib

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("module", [
    "hcatostialocalization.train.trainer",
    "hcatostialocalization.loss.loss_factory",
    "hcatostialocalization.data.dataset.General_Class_Dataset_Multi",
    "hcatostialocalization.logger.logger",
])
def test_import(module):
    importlib.import_module(module)
