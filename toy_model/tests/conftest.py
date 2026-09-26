"""Put training/ on sys.path so tests import the package modules directly.

The training code uses flat imports (`import matrices`, `from utils import ...`)
because Hydra runs it with its own directory on sys.path. Tests mirror that.
"""

import os
import sys

# Data generation in these tests is tiny; keep JAX off the GPU so the suite can
# run alongside a training job.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
