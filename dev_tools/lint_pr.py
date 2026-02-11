#!/usr/bin/env python3
"""Lint a Pull Request title to ensure it follows Conventional Commits.

Usage:
    python lint_pr.py "<PR_TITLE>"

The title must follow the format:
    <type>(<scope>): <Description>

Where:
    <type> is one of: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test
    <scope> is optional, lowercase alphanumeric with hyphens
    <Description> starts with a capital letter
"""
import re
import sys

# Regex pattern for Conventional Commits with capitalized description
# Group 1: Type
# Group 2: Optional Scope (including parens)
# Match: ": " followed by Capital letter and then anything
PATTERN = (
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([a-z0-9-]+\))?: [A-Z].+$"
)


def main() -> None:
    """Validate the PR title from command-line arguments."""
    if len(sys.argv) < 2:
        print("Error: No PR title provided.")
        sys.exit(1)

    title = sys.argv[1]
    if not re.match(PATTERN, title):
        print(f"Error: PR title '{title}' does not follow Conventional Commits.")
        print("Format: <type>(<scope>): <Description>")
        print("  - Types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test")
        print("  - Scope: Optional, lowercase alphanumeric with hyphens (e.g., (scope))")
        print("  - Description: Must start with a capital letter")
        sys.exit(1)

    print("Success: PR title is valid.")


if __name__ == "__main__":
    main()
