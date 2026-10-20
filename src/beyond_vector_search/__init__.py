"""Agentic document search over a folder of files.

The model lists, greps, and reads the live documents, then follows
cross-references. Nothing is chunked or embedded unless you opt into
the hybrid semantic tool.
"""

__version__ = "0.1.0"

from beyond_vector_search.agent import ask
from beyond_vector_search.config import Settings
from beyond_vector_search.result import Result

__all__ = ["Result", "Settings", "ask", "__version__"]
