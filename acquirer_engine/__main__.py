"""`python -m acquirer_engine`: version guard, then the CLI."""

import sys

# Checked before any import: older interpreters fail deep inside pydantic with an unreadable traceback.
if sys.version_info < (3, 10):  # noqa: UP036 - deliberate guard for users on an older interpreter
    sys.exit(
        f"error: acquirer_engine needs Python 3.10+, but this is {sys.version.split()[0]} "
        f"({sys.executable}). Create the venv with python3.10 or newer."
    )

from .cli import main

sys.exit(main())
