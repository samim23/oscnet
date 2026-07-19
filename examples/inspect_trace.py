"""Inspect an OscNet NPZ trace into coupling / field / synchrony panels.

Examples
--------
```bash
python examples/inspect_trace.py \\
  outputs/smoke/fashion_mnist_generator_horn/traces/mnist_generator_trace_epoch_001.npz

python examples/inspect_trace.py \\
  outputs/reference/mnist_winfree_field_smoke \\
  --grid-shape 4 4

python examples/inspect_trace.py path/to/run --all-traces -o /tmp/inspect_out
```
"""

from oscnet.inspection import main

if __name__ == "__main__":
    raise SystemExit(main())
