# Fractal Coupling for Oscillatory Networks

Hierarchical fractal coupling that improves memory tasks in oscillatory neural networks.

## Key Finding

**Hierarchical fractal coupling (depth=1) dramatically improves memory tasks:**

- **+131% improvement** in associative recall
- **+271% improvement** in sequential binding  
- **+13-22% improvement** in few-shot learning

The advantage is **structural**: discrete scales create clear boundaries that separate patterns better than dense coupling.

## Quick Start

```bash
# Run unit tests
pytest tests/test_fractal_coupling.py -v

# Run integration example
python examples/fractal/code/integration_example.py
```

## Usage

```python
from oscnet.core import HierarchicalCouplingLayer

# Default optimal configuration (depth=1, strength=0.5)
h2h = HierarchicalCouplingLayer(hidden_dim=64, key=key)

# Or specify explicitly
h2h = HierarchicalCouplingLayer(hidden_dim=64, depth=1, inter_block_strength=0.5, key=key)
```

### Adaptive Variant (for overlapping patterns)

```python
from oscnet.core import AdaptiveFractalCouplingLayer

# Learns to route patterns to different fractal scales
h2h = AdaptiveFractalCouplingLayer(hidden_dim=64, key=key)
```

## Optimal Configuration

| Parameter | Optimal | Notes |
|-----------|---------|-------|
| `depth` | 1 | Shallow nesting works best |
| `inter_block_strength` | 0.5 | Moderate coupling between blocks |

**Important**: Deeper is NOT better. Depth=1 outperforms depth=2 or 3.

## When to Use

**Best for:**
- Associative recall / pattern completion
- Content-addressable memory
- Few-shot learning
- Sequential pattern binding

**Not ideal for:**
- Overlapping patterns (use AdaptiveFractalCouplingLayer)
- Classification tasks (minimal advantage)

## Why It Works

Fractal coupling's advantage is **structural, not wave dynamics**:

1. Discrete scales create clear pattern boundaries
2. Hierarchical blocks outperform power-law coupling
3. The topology organizes memory attractors

See `code/integration_example.py` for usage.
