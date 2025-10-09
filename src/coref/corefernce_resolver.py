try:
    from fastcoref import LingMessCoref
except ImportError:
    raise ImportError(
        "fastcoref is not installed. Install it via `pip install fastcoref`."
    )

from src.coref.coreference_flow import coreference

class CoreferenceResolver:
    """
    Coreference resolver using LingMessCoref via your coreference_implementor.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize LingMessCoref with attention fix.

        Args:
            device (str): Device to run on, "cpu" or "cuda".
        """
        self.model = LingMessCoref(device=device)

    def resolve(self, text: str) -> str:
        """
        Resolve coreferences in the given text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with coreferences resolved.
        """

        return coreference(text)
