# Deep Learning with Python

A hands-on learning repository for building deep learning knowledge from the ground up, starting with NumPy fundamentals. Each lesson is a self-contained Python script with detailed comments explaining the concepts.

## Prerequisites

- Python 3.12
- NumPy (`pip install numpy`)

## Repository Structure

Lessons are organized by week and day:

```
week01/                          # NumPy fundamentals
├── day01_numpy_basics.py        # Dot product: naive loop vs np.dot() performance
└── day01_numpy_broadcasting.py  # Broadcasting rules, shapes, and practical examples
```

## Getting Started

Clone the repository and run any lesson directly:

```bash
git clone <repo-url>
cd dlwp
pip install numpy
python week01/day01_numpy_basics.py
```

## Curriculum

### Week 1 — NumPy Fundamentals

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | [NumPy Basics](week01/day01_numpy_basics.py) | Vectorized operations, `np.dot()` vs manual loops, performance comparison |
| 1 | [Broadcasting](week01/day01_numpy_broadcasting.py) | Broadcasting rules, shape compatibility, data centering, image normalization |

## Code Style

- Formatted with [Black](https://github.com/psf/black)
- Heavy inline comments — each file is meant to teach, not just run

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
