# Contributing to This Repository

We value clarity, consistency, and usability in our scripts. Please adhere to the following principles when contributing:


## 🧱 Code Structure

- Scripts **must be modular**, with a clearly defined `main()` function.
- Include a clear **configuration section at the top** of each script.
  - Prefer inline variable configuration over `argparse`.
- Use intuitive success messages at the end of script execution.
  - e.g., `print("Script completed successfully.")` or equivalent `logging` call.
- **Do not import shared helper functions** from external modules like `gtfs_helpers.py`.
  - Instead, **reproduce the minimal helper function directly in the script** where it's used.
  - This makes each script self-contained and easier for beginners to read, copy, and modify without needing to navigate the whole repo.

## ⚙️ Runtime Behavior

- Prefer the `logging` module over `print()` for diagnostics or warnings.
- Implement **graceful, actionable error handling** — no cryptic tracebacks.
- Use placeholder filenames that are clean, minimal, and safe to run (e.g., no real paths or usernames).
- Default to:
  - **Washington, DC CRS** unless otherwise noted.
  - **Imperial units** (feet/miles), with metric options clearly noted when used.

## 🧪 Testing & Review

- **Manual testing is required** before committing. There are no automated tests (yet).
- All commits must use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## 🧼 Code Style

Most formatting and style issues are enforced automatically via **Ruff** and checked in CI. This includes:

- PEP 8 compliance
- Import order
- Blank lines, indentation, etc.
- Google-style docstrings (via `pydocstyle`)
- Avoiding use of `print()` (warned via `T201`)

You do **not** need to manually run linters before committing, but you should review CI feedback if it fails.

## 📁 File Organization

- Add new scripts to the appropriate subfolder within `scripts/`, based on function (e.g., `ridership_tools/`, `gtfs_exports/`).
- Any reusable helper functions should be added to `helpers/gtfs_helpers.py`, not copied into individual scripts.

---

For further details, contact the repository maintainer.
