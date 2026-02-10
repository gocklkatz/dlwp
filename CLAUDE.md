# CLAUDE.md

## Project Overview

Educational Python repository for learning deep learning fundamentals, starting with NumPy. Lessons are organized by week and day. Licensed under GNU GPLv3.

## Repository Structure

```
dlwp/
├── LICENSE                          # GNU GPLv3
├── CLAUDE.md                        # This file
├── .idea/                           # PyCharm/IntelliJ IDE config
└── week01/                          # Week 1: NumPy fundamentals
    ├── day01_numpy_basics.py        # Dot product performance comparison
    └── day01_numpy_broadcasting.py  # Broadcasting rules and examples
```

Lessons follow a `weekNN/dayNN_<topic>.py` naming convention.

## Development Environment

- **Python version**: 3.12
- **Key dependency**: NumPy (install with `pip install numpy`)
- **Formatter**: Black (configured in IDE settings)
- **IDE**: PyCharm/IntelliJ IDEA

There is no `requirements.txt`, `pyproject.toml`, or `setup.py`. Dependencies must be installed manually.

## Running Code

Scripts are standalone educational files, not importable modules. Run them directly:

```bash
python week01/day01_numpy_basics.py
python week01/day01_numpy_broadcasting.py
```

## Testing

No test framework is configured. There are no tests.

## CI/CD

No CI/CD pipelines are configured.

## Conventions

- **File naming**: `dayNN_topic_name.py` inside `weekNN/` directories
- **Code style**: Black formatter, heavy inline comments explaining concepts
- **Structure**: Each file is a self-contained lesson script with examples and print output
- **Comments**: Serve as teaching material — explain the "why" not just the "what"

## Notes for AI Assistants

- This is a learning project; prioritize clarity and educational value in code
- Keep files self-contained and runnable as standalone scripts
- Use detailed inline comments to explain concepts
- Follow the existing week/day organizational pattern when adding new content
- Format code with Black
