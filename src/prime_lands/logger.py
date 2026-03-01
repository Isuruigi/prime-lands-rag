"""
Structured logging using loguru.
Every module gets a context-tagged logger.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(level: str = "INFO", serialize: bool = False) -> None:
    """
    Configure application-wide structured logging.

    Args:
        level: Minimum log level (DEBUG/INFO/WARNING/ERROR)
        serialize: If True, output JSON (for log aggregators like Datadog)
    """
    logger.remove()
    
    # Console output with colors
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        serialize=serialize,
        backtrace=True,
        diagnose=True,
    )
    
    # File output with JSON serialization
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "prime_lands_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        serialize=True,   # JSON logs for file
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str):
    """
    Get a named logger for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance bound to module context
    """
    return logger.bind(module=name)
