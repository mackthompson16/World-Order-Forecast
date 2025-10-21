from pathlib import Path
import sys

# Deprecated monolith: this file now delegates to the modular pipeline.
# Use `python scripts/run_pipeline.py` for a stable entrypoint.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import run


def main():
    print("[DEPRECATED] scripts/build_empire_standings.py is deprecated. Delegating to src.pipeline.orchestrator.run().")
    run()


if __name__ == "__main__":
    main()

