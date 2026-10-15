import os
import socket
from omegaconf import OmegaConf

def check_environment():
    # 1. Detect System Info
    hostname = socket.gethostname()
    env_hostname = os.environ.get("HOSTNAME")
    env_computername = os.environ.get("COMPUTERNAME")
    
    print("="*50)
    print(" PROJECT CONFIGURATION DIAGNOSTIC ")
    print("="*50)
    print(f"OS Node Name (socket): {hostname}")
    print(f"Env 'HOSTNAME':        {env_hostname}")
    print(f"Env 'COMPUTERNAME':    {env_computername}")
    print("-" * 50)

    # 2. Try to load and resolve the machine config
    try:
        # Find machine.yaml in the current directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        machine_cfg_path = os.path.join(base_path, "machine.yaml")
        
        if not os.path.exists(machine_cfg_path):
            print(f" ERROR: Could not find '{machine_cfg_path}'")
            return

        cfg = OmegaConf.load(machine_cfg_path)
        OmegaConf.resolve(cfg)

        detected_host = cfg.host_settings.current_host
        
        print(f" SUCCESS: Machine config loaded.")
        print(f" Detected Identity:  {detected_host}")
        
        if detected_host == "default_user":
            print("\n  WARNING: Using 'default_user' fallback.")
            print(f"   To fix this, add an entry for '{hostname}' in machine.yaml.")
        else:
            print(f"\n🚀 Welcome, {detected_host}!")
            print(f"   Data Root:  {cfg.active_paths.data_root}")
            print(f"   Output Dir: {cfg.active_paths.output_dir}")

    except Exception as e:
        print(f" ERROR: Resolution failed.")
        print(f" Details: {e}")
    
    print("="*50)

if __name__ == "__main__":
    check_environment()