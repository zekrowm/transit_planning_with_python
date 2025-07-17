# Contributing to This Repository

This project is built for transit planners, transit analysts, and civic technologists who want readable,
self-contained Python scripts for transportation planning. We value clarity, consistency, and usability
in our scripts to make them usable by a wider audience. Please follow the following principles when contributing:


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
- Use placeholder filenames that are clean, minimal, and safe to run (e.g., r"Path\\To\\Output_Folder", "input_data.csv").
- Default to:
  - **Washington, DC CRS** unless otherwise noted.
  - **Imperial units** (feet/miles), with metric options clearly noted when used.

## 🧪 Testing & Review

- **Manual testing is required** before submitting a pull request. There are no automated tests yet.
- All commits must use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for clear project history.
- All pull requests are automatically tested for:
  - Style and formatting using `ruff`.
  - Static typing using `ty`.
- You **do not** need to run linters or type checkers manually, but you **must fix** any issues flagged by the CI pipeline before requesting a review.

## 🧼 Code Style

This project uses `ruff` to enforce formatting, linting, and docstring style, and `ty` for non-blocking type checks.

Most formatting issues (indentation, line length, spacing) are auto-corrected by Ruff on PRs.

- The following are enforced in CI:
  - PEP 8 layout and formatting
  - The enforced line length is **100 characters**
  - Google-style docstrings
  - Consistent import ordering (`isort`-compatible)
  - Type annotations (with some leniency for `Any`)
  - Avoiding `print()` (via Ruff rule `T201`) — use `logging` instead        

**Note:** Ruff auto-fixes are pushed back to your PR branch automatically by the GitHub Actions workflow.

## 📁 File Organization

- Add new scripts to the appropriate subfolder within `scripts/`, based on function (e.g., `ridership_tools/`, `gtfs_exports/`).
- Any reusable helper functions should be added to `helpers`, and copied into individual scripts.

---

For further details, contact the repository maintainer.
