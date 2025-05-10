# Contributing to This Repository

We value clarity, consistency, and ease-of-use in our scripts. Please adhere to the following principles when contributing:

## Repository Principles

1. **Usability and interpretability for new users are paramount.**
2. **Adhere strictly to PEP-8 standards**
   - Maximum line length: 100 characters.
3. **Scripts must be modular, using a clearly defined `main()` function.**
4. **Use clear, minimal, and meaningful comments/docstrings.**
5. **Follow standardized module docstring formatting.**
6. **Implement graceful, actionable error handling.**
7. **Prefer simple `print()` statements over logging.**
8. **Include intuitive configuration sections at the beginning of scripts.**
   - Prefer inline configuration over command-line arguments (`argparse`).
9. **Clearly print 'script has run successfully' messages at completion.**
10. **Ensure sanitized and standardized filenames.**
11. **Commit messages must follow Conventional Commit style.**
12. **Maintain Pylint score minimum: 8.0 (target: 9.0). Do not silence warnings.**
13. **Run manual QA/QC checks using Black and Pylint prior to commits.**
14. **Perform manual testing before committing (no automated unit tests required).**
15. **Use standardized utility functions from `gtfs_helpers.py`.**
16. **Use Washington DC default CRS unless otherwise noted.**
17. **Default units: Imperial (feet/miles); metric options should be clearly available.**

---

For further details, contact the repository maintainer.
