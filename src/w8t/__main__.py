"""Local entry point, Django-style:

    python -m w8t            # starts the app (streamlit run src/w8t/app/app.py)
    python -m w8t migrate    # applies database migrations (alembic upgrade head)

Extra arguments after ``python -m w8t`` go to Streamlit, e.g. ``python -m w8t --server.port 8600``.
Must be run from the project root (where ``alembic.ini`` and ``.env`` live).
"""

from __future__ import annotations

import sys
from pathlib import Path

ENTRY = Path(__file__).parent / "app" / "app.py"


def main(argv: list[str]) -> int:
    if argv[:1] == ["migrate"]:
        from alembic.config import main as alembic_main

        alembic_main(argv=["upgrade", "head"])
        return 0

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(ENTRY), *argv]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
