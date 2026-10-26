"""Run data preparation, then exit after all pipeline files are committed."""
from pathlib import Path
import os
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_40b


if __name__ == "__main__":
    exit_code = 0
    try:
        run_40b.main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        elif exc.code is not None:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1

    # main() has unwound its file/lock contexts. Successful preparation and
    # graceful stops explicitly flush and fsync token files and atomic state.
    # PyArrow 25.0.1 still crashes during interpreter finalization on this
    # host; gc.collect() alone did not fix it. Exit only after main returns
    # (or its failure is reported), so native finalizers cannot race Python.
    # Failures stay nonzero, allowing Supervisor to resume committed state.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
