import os
import glob
import socket
from omegaconf import OmegaConf, SCMode

# Make hostname-based YAML resolution work even when the shell variable exists
# but is not exported to Python's process environment.
raw_hostname = socket.getfqdn() or socket.gethostname()
os.environ.setdefault("HOSTNAME", raw_hostname.split('.')[0])

# Get the directory where this script is located
_this_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Initialize an empty OmegaConf object
config = OmegaConf.create()

# 2. Find all .yaml files recursively from _this_dir downwards
# The recursive=True flag is essential here
search_pattern = os.path.join(_this_dir, "**", "*.yaml")
all_yaml_files = sorted(glob.glob(search_pattern, recursive=True))

for path in all_yaml_files:
    # Load the current file
    cfg = OmegaConf.load(path)
    # Merge it into the main config object
    config = OmegaConf.merge(config, cfg)

# 2b. Experiment inheritance: an experiment with `extends: <other experiment>` starts from a
# deep copy of the parent and overrides only the fields it defines (dicts are merged key by
# key, lists and scalars are replaced). Chains (A extends B extends C) are supported.
def _resolve_experiment_inheritance(cfg):
    experiments = cfg.get("experiments")
    if not experiments:
        return cfg

    resolved = {}

    def _resolve(name, chain=()):
        if name in resolved:
            return resolved[name]
        if name in chain:
            raise ValueError(f"Circular 'extends' in experiments: {' -> '.join(chain + (name,))}")
        if name not in experiments:
            raise ValueError(f"Experiment '{chain[-1]}' extends unknown experiment '{name}'")
        node = experiments[name]
        parent = node.get("extends")
        if parent is not None:
            node = OmegaConf.merge(_resolve(parent, chain + (name,)), node)
        resolved[name] = node
        return node

    for name in list(experiments.keys()):
        _resolve(name)
    for name, node in resolved.items():
        cfg.experiments[name] = node
    return cfg


config = _resolve_experiment_inheritance(config)

# 3. Resolve interpolations (like ${some.key})
OmegaConf.resolve(config)

# 4. Convert to a standard Python dictionary
configd = OmegaConf.to_container(
    config, 
    resolve=True, 
    throw_on_missing=True, 
    structured_config_mode=SCMode.INSTANTIATE
)

# TEST
if __name__ == "__main__":
    import pprint
    print(f"--- Found and merged {len(all_yaml_files)} YAML files ---")
    for f in all_yaml_files:
        print(f"Loaded: {os.path.relpath(f, _this_dir)}")
    print("-" * 40)
    pprint.pprint(configd)

    print("-" * 40)
    print("Suggestion: run this test with `python .....config | less` to view it better.")
