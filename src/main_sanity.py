# src/main_sanity.py

from pathlib import Path

from utils.config import load_config
from utils.logging import setup_logging
from utils.seed import set_seed


def main():
    # 1. Setup logging
    logger = setup_logging(name="sanity", log_dir="experiments/logs", reset_handlers=True)

    # 2. Load config
    cfg_path = Path("configs/main_cnn_template.yaml")
    cfg = load_config(cfg_path)
    logger.info(f"Loaded config from: {cfg.path}")
    logger.info("Config content:\n" + cfg.pretty())

    # 3. Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    logger.info("Sanity check completed successfully.")


if __name__ == "__main__":
    main()
