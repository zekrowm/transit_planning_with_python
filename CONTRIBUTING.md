# Contributing to This Repository

We value clarity, consistency, and ease-of-use in our scripts. Please adhere to the following principles when contributing:

## Repository Principles

1. **Usability and interpretability for new users are paramount.**
2. **Adhere strictly to PEP-8 standards**
3. **Scripts must be modular, using a clearly defined `main()` function.**
4. **Use clear, minimal, and meaningful comments.**
5. **Follow Google docstring formatting style.**
6. **Implement graceful, actionable error handling.**
7. **Prefer logging over `print()` statements.**
8. **Include intuitive configuration sections at the beginning of scripts.**
   - Prefer inline configuration over command-line arguments (`argparse`).
9. **Clearly log or print 'script has run successfully' messages at completion.**
10. **Ensure sanitized and standardized placeholder filenames.**
11. **Commit messages must follow Conventional Commits style.**
12. **Use pre-commit linting with Ruff to maintain minimum standards.**
13. **Perform manual testing before committing (no automated tests required).**
14. **Periodically run static checks to identify complex issues and address as needed.**
15. **Use standardized utility functions from `gtfs_helpers.py`.**
16. **Use Washington DC default CRS unless otherwise noted.**
17. **Default units: Imperial (feet/miles); metric options should be clearly available.**

---

For further details, contact the repository maintainer.
