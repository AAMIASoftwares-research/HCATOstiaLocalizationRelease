import pathlib

import yaml

from helpers import RELEASED_MODELS


def test_config_resolves():
    from hcatostialocalization.config.config import config
    for key in ("root_repo", "output_dir", "data_root"):
        assert config.active_paths.get(key), f"active_paths.{key} missing (config/machine.yaml)"


def test_split_files_exist_and_sizes():
    from hcatostialocalization.config.config import config
    expected_test = {"ImageCAS": 50, "ASOCA": 4, "CAT08": 4}   # the 58-volume test set of the paper
    for name, n_test in expected_test.items():
        path = pathlib.Path(config.dataset[name].yaml_path)
        assert path.is_file(), path
        split = yaml.safe_load(path.read_text())[name]["split"]
        assert len(split["test"]) == n_test
        assert len(split["train"]) > 0 and len(split["val"]) > 0
        assert {"image", "label"} <= set(split["train"][0])


def test_hub_lists_released_models_only():
    from hcatostialocalization.model.hub.hub import models_hub_pretrained
    assert sorted(models_hub_pretrained) == sorted(RELEASED_MODELS)
