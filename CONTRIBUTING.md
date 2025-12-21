# Contributing to This Repository

This project is built for transit planners, transit analysts, and civic technologists who want readable,
self-contained Python scripts for transportation planning. We value clarity, consistency, and usability
in our scripts to make them usable by a wider audience. Please follow these principles when contributing:

## 👥 How to Contribute

Participation is welcome from anyone, whether you’re new to coding, an experienced GitHub user, or a seasoned developer:
- **Beginners:**  
  - Feel free to copy, modify, and use scripts without any expectation of interaction.
- **Intermediate Users:**  
  - Create a GitHub Issue to report bugs, request new features, or ask questions.
  - Clearly describe your issue, including error messages, expected vs. actual results, and steps to reproduce the issue.
- **Advanced Users:**  
  - Submit Pull Requests (PRs) with proposed improvements or fixes.
  - Follow the instructions below to ensure your PRs meet project standards.

## 🧱 Code Structure

- Scripts **must be modular**, with a clearly defined `main()` function.
- Include a clear **configuration section at the top** of each script.
  - Prefer inline variable configuration over `argparse`.
- Use intuitive success messages at the end of script execution.
  - e.g., `print("Script completed successfully.")` or equivalent `logging` call.
- Do **not import** functions from the shared `helpers/` directory at runtime.
  - Instead, **copy the relevant helper functions** into your script.
  - This keeps each script self-contained and easier for beginners to understand, run, and modify.
- The `helpers/` directory holds the **canonical version** of shared functions. Any differences between a script’s local copy and the canonical version will be flagged in CI.

## ⚙️ Runtime Behavior

- Prefer the `logging` module over `print()` for diagnostics or warnings.
- Implement **graceful, actionable error handling** — no cryptic tracebacks.
- Use placeholder filenames that are clean, minimal, and safe to run (e.g., r"Path\\To\\Output_Folder", "input_data.csv").
- Default to:
  - **Washington, DC CRS** unless otherwise noted.
  - **Imperial units** (feet/miles), with metric options clearly noted when used.

## 🧪 Testing & Review

- All commits and **Pull Request titles** must use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for clear project history.
- All pull requests are automatically tested for:
  - Style and formatting using `ruff`.
  - Static typing using `ty`.
  - Unit tests (where present) using `pytest`.
- You do **not** need to run linters or type checkers manually, but you **must fix** any issues flagged by the CI pipeline before requesting a review.

**Manual testing policy (scripts):**  
New or modified scripts under `scripts/` **must be manually tested** before opening a PR. See the checklist below.

**Manual test checklist (scripts):**
- [ ] Script runs end-to-end with clean console output (or appropriate logging).
- [ ] Configuration section at the top is clear and minimal; defaults produce a safe no-op or sample run.
- [ ] Input/output paths are valid; exported files are created with expected names and sizes.
- [ ] Error messages are actionable (no cryptic tracebacks for expected user mistakes).
- [ ] Runtime is reasonable on a small sample dataset; no hidden network or large temporary files.

---

### 🧩 Unit Tests for Helper Functions

If you add or significantly modify a function in the `helpers/` directory:

- Write a **unit test** that exercises its normal behavior and at least one error condition.
- Save new tests under `tests/unit/` following the naming pattern `test_<module>.py`.
- Use small, synthetic input data—do **not** rely on external files or network access.
- Tests should run quickly (<1 s each) and be deterministic.
- These tests protect against silent failures caused by future changes to dependencies (e.g., pandas, geopandas).
- A pull request adding or modifying helpers **without** a corresponding test may be asked to add one before review.

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
- If you create a helper that’s reused across multiple scripts:
  - Add the canonical version to the appropriate file under `helpers/`.
  - Then copy that helper into any script that uses it.
- Do **not** import functions from one script into another or from `helpers/` at runtime.

## 🌳 GitHub Contribution Workflow

Follow these instructions when contributing code via GitHub:

1. **Fork the repository.**
   - Click the "Fork" button on GitHub to create your own copy.
2. **Clone the repository locally.**
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
   cd <REPO-NAME>
   ```
3. **Create a feature branch.**
   ```bash
   git checkout -b feature/your-descriptive-feature-name
   ```
4. **Make your changes.**
   - Keep commits small and clearly described with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. **Push your branch to your fork.**
   ```bash
   git push -u origin feature/your-descriptive-feature-name
   ```
6. **Open a Pull Request.**
   - Title your Pull Request using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
   - Clearly describe your changes, referencing any related issues.
7. **Respond to feedback.**
   - Update your PR with suggested changes until your contribution is approved.

---

For further details, contact the repository maintainer.
