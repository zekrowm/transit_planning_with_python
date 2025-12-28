import json
import logging
import os
import re
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Regex for Conventional Commits
# https://www.conventionalcommits.org/en/v1.0.0/
# Pattern: type(scope): description
# Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
CONVENTIONAL_COMMIT_PATTERN = r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9-]+\))?(!)?: .+$"

def validate_message(message: str, context: str) -> bool:
    """Validates a message against the Conventional Commits pattern."""
    if not re.match(CONVENTIONAL_COMMIT_PATTERN, message):
        logger.error(f"{context} does not follow Conventional Commits.")
        logger.info(f"  Current: '{message}'")
        logger.info("  Expected format: type(scope): description")
        logger.info("  Example: feat(ui): add dark mode button")
        logger.info("  Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert")
        return False
    return True

def get_commits_from_git(base_ref: str):
    """Retrieves commit messages from git log."""
    # Fetch the base ref to ensure we can compare
    try:
        # Check if we are in a shallow clone, if so, we might fail if we don't have the base
        # But assuming fetch-depth: 0 in CI
        # We use --no-merges to avoid validating the merge commit itself in PRs
        cmd = ["git", "log", f"origin/{base_ref}..HEAD", "--no-merges", "--format=%s"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = result.stdout.strip().split("\n")
        # Filter out empty strings if any
        return [c for c in commits if c]
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get commits via git: {e}")
        return []

def main():
    has_errors = False

    # 1. Validate PR Metadata (Title and Body)
    # This relies on the GitHub Actions event payload
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        logger.info(f"Loading event data from {event_path}")
        try:
            with open(event_path, "r") as f:
                event = json.load(f)

            # PR Title
            pr = event.get("pull_request")
            if pr:
                title = pr.get("title", "")
                body = pr.get("body", "")

                logger.info("Validating PR Title...")
                if not validate_message(title, "PR Title"):
                    has_errors = True
                else:
                    logger.info("PR Title is valid.")

                logger.info("Validating PR Body...")
                if not body or len(body.strip()) < 10:
                    logger.error("PR Body is too short or empty.")
                    logger.info("  Please provide a description of the changes (at least 10 characters).")
                    has_errors = True
                else:
                    logger.info("PR Body is valid.")
            else:
                logger.info("Not a Pull Request event. Skipping PR metadata validation.")

        except Exception as e:
            logger.error(f"Error parsing event payload: {e}")
            has_errors = True
    else:
        logger.warning("GITHUB_EVENT_PATH not found. Skipping PR metadata validation (Title/Body).")

    # 2. Validate Commit Messages
    # In a PR, we want to check commits introduced by this PR.
    # GITHUB_BASE_REF is set in PR workflows to the target branch (e.g. main)
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if not base_ref:
        # Fallback for local testing or non-PR events
        logger.info("GITHUB_BASE_REF not set. Defaulting to 'main'.")
        base_ref = "main"

    logger.info(f"Validating commits against base '{base_ref}'...")
    commits = get_commits_from_git(base_ref)

    if not commits:
        logger.warning("No commits found to validate (or git failed).")
    else:
        for commit_msg in commits:
            if not validate_message(commit_msg, f"Commit '{commit_msg}'"):
                has_errors = True

    if has_errors:
        sys.exit(1)

    logger.info("All checks passed!")

if __name__ == "__main__":
    main()
