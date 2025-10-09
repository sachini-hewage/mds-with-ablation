import json
import subprocess
from pathlib import Path

from src.summariser.prompt_templates import (
    PAIRING_TEMPLATE,
    SENTENCE_CLUSTER_TEMPLATE,
    PARAGRAPH_CLUSTER_TEMPLATE,
)

class Summariser:
    """
    Summariser class to generate summaries using an Ollama LLM.

    Supports three summarization methods:
      1. pairing
      2. sentence_clustering
      3. paragraph_clustering

    Uses pre-defined prompt templates for each method and calls the specified Ollama model.
    """

    def __init__(self, model="qwen3:8b"):
        """
        Initialize Summariser.

        Args:
            model (str): Ollama model name to use for summarization.
        """
        self.model = model


    # Core entrypoint to summarisation

    def summarize(self, data, method):
        """
        Generate summary for given data using the selected method.

        Args:
            data (dict/list): Input data to summarize (e.g., paired paragraphs, clustered sentences, etc.)
            method (str): Summarization method: "pairing", "sentence_clustering", or "paragraph_clustering"

        Returns:
            str: Generated summary from the Ollama model
        """
        # Select template based on method
        if method == "pairing":
            template = PAIRING_TEMPLATE
        elif method == "sentence_clustering":
            template = SENTENCE_CLUSTER_TEMPLATE
        elif method == "paragraph_clustering":
            template = PARAGRAPH_CLUSTER_TEMPLATE
        else:
            raise ValueError(f"Unknown summarization method: {method}")

        # Combine template with JSON-formatted input data
        input_text = template + "\n\nData:\n" + json.dumps(data, indent=2, ensure_ascii=False)

        # Call LLM with constructed prompt
        return self.call_llm(input_text)


    # Generic Ollama call

    def call_llm(self, prompt):
        """
        Run Ollama LLM with given prompt.

        Args:
            prompt (str): Full text prompt to feed into the model

        Returns:
            str: Model output (summary text)
        """
        result = subprocess.run(
            ["ollama", "run", self.model, "--hidethinking"], # In case I use a thinking model
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Decode stdout
        output = result.stdout.decode("utf-8").strip()

        # Print errors if LLM call failed
        if result.returncode != 0:
            print("Ollama stderr:", result.stderr.decode("utf-8"))

        return output


    # Ablation wrapper
    def run_for_ablation(self, mode, method, input_file, results_dir):
        """
        Wrapper to generate and save summaries for ablation experiments.

        Args:
            mode (str): Mode of the experiment (e.g., dataset split or ablation type)
            method (str): Summarization method
            input_file (str/Path): JSON file containing input data
            results_dir (str/Path): Directory to save generated summaries
        """
        print(f"[Summariser] Summarizing {method} results for mode={mode}")

        # Load input data from JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Generate summary
        summary = self.summarize(data, method)

        # Save summary to output file
        output_file = Path(results_dir) / f"summary_{method}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"[Summariser] Saved {method} summary to {output_file}")
