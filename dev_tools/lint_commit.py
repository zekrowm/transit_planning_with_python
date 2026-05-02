#!/usr/bin/env python3
"""Lint a commit message subject line to ensure it follows Conventional Commits.

Usage:
    python lint_commit.py <commit-msg-file>

The subject line (first line) must follow the format:
    <type>(<scope>): <description>

Where:
    <type> is one of: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test
    <scope> is optional, lowercase alphanumeric with hyphens
    <description> starts with a lowercase letter

Auto-generated subjects (Merge …, Revert "…") are skipped.
"""

import re
import sys

PATTERN = (
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([a-z0-9-]+\))?: [a-z].+$"
)

SKIP_PATTERN = r"^(Merge |Revert \")"


def main() -> None:
    """Validate the commit message subject from the file path argument."""
    if len(sys.argv) < 2:
        print("Error: No commit message file provided.")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        subject = f.readline().rstrip("\n")

    if re.match(SKIP_PATTERN, subject):
        print(f"Skipped: auto-generated commit '{subject}'.")
        sys.exit(0)

    if not re.match(PATTERN, subject):
        print(f"Error: Commit subject '{subject}' does not follow Conventional Commits.")
        print("Format: <type>(<scope>): <description>")
        print("  - Types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test")
        print("  - Scope: Optional, lowercase alphanumeric with hyphens (e.g., (scope))")
        print("  - Description: Must start with a lowercase letter")
        sys.exit(1)

    print("Success: Commit subject is valid.")


if __name__ == "__main__":
    main()
