"""Run the centralized baseline with the shared experiment CLI."""

import sys
from run_experiment import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "centralized"]
    main()
