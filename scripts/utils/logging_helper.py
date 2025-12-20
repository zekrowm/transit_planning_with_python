"""A basic logging helper."""
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
