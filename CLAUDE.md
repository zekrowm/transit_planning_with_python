# Instructions for Claude Code

## PR titles (REQUIRED — enforced in CI)
Every PR title MUST follow Conventional Commits with a lowercase
description. This is checked by `dev_tools/lint_pr.py` and will
fail CI otherwise.

Format: `<type>(<optional-scope>): <description>`

- Allowed types: build, chore, ci, docs, feat, fix, perf,
  refactor, revert, style, test
- Scope (optional): lowercase letters, digits, hyphens
- Description: starts with a lowercase letter, no period at end

Good examples:
  feat(gtfs): add stop spacing validator
  fix(utils): handle empty shapes.txt gracefully
  docs: clarify pr title convention

Bad examples (will fail CI):
  Change PR title convention to lowercase description   (no type)
  feat: Add stop spacing validator                      (capitalized)
  feat(GTFS): add stop spacing validator                (uppercase scope)

Before opening a PR, double-check the title matches this format.

## Commit messages
Follow the same Conventional Commits format for commits. Not
enforced in CI, but kept consistent for project history.

## Other conventions
See CONTRIBUTING.md for code style, testing policy, and the
file-organization rules — especially the "copy helpers from
utils/, don't import them at runtime" rule.
