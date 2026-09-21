"""Compatibility entry point for IID experiments; accepts shared CLI options."""

import sys
from run_experiment import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--partition", "iid"]
    main()

