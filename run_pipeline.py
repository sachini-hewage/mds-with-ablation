import warnings

# Suppress FutureWarnings (only for this block)
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
from src.pipeline import Pipeline

if __name__ == "__main__":
    with open("configs/clustering_pairing.yaml") as f:
        config = yaml.safe_load(f)

    # Pass config to the Pipeline
    pipeline = Pipeline(config)
    pipeline.run()


