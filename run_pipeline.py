import warnings
import shutil
from pathlib import Path
import yaml
from src.pipeline import Pipeline

# Suppress FutureWarnings (only for this block)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":

    # Path to processed, results folders and all_results file
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    results_dir = data_dir / "results"
    all_results_file = Path("all_results.json")

    # Initial cleanup before any runs
    for folder in [processed_dir, results_dir]:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"\nCleaned up {folder}")

    if all_results_file.exists():
        all_results_file.unlink()
        print(f"\nDeleted {all_results_file}")

    # Load config
    with open("configs/clustering_pairing.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Run for first 10 instances
    for instance_id in range(27, 28):
        print(f"\n=== Running pipeline for instance {instance_id} ===")

        # Run pipeline for current instance
        pipeline.run(instance_id)

        # Clean up data folder after each run
        # for folder in [processed_dir, results_dir]:
        #     if folder.exists():
        #         shutil.rmtree(folder)
        #         print(f"Cleaned up {folder}")

    print("\n=== Finished all instances ===")
