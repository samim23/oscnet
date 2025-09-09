"""
Logging configuration and setup utilities.

This module provides utilities for setting up application logging with
configurable levels, file output, and JAX-specific optimizations.
"""

import logging
import jax
from typing import Optional


def setup_application_logger(
    name: str = "oscnet",
    level: int = logging.INFO, 
    enable_file_logging: bool = False, 
    log_file: str = "training.log"
) -> logging.Logger:
    """
    Setup application logger with configurable levels and optional file logging.
    
    Args:
        name: Logger name (default: "oscnet")
        level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        enable_file_logging: Whether to also log to a file
        log_file: Path to log file if file logging is enabled
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # CRITICAL: Clear ALL handlers and disable propagation to prevent duplicates
    logger.handlers.clear()
    logger.propagate = False  # Prevent propagation to root logger
    
    # Set logger level
    logger.setLevel(level)
    
    # Console handler - ONLY ONE
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Device detection messages
    try:
        available_platforms = [d.device_kind for d in jax.devices()]
        if 'gpu' in available_platforms:
            logger.info("🚀 GPU detected and configured")
        else:
            logger.info("💻 Using CPU (no GPU detected)")
    except Exception as e:
        logger.warning(f"⚠️  Device detection: {e}")
        logger.info("💻 Defaulting to CPU")
    
    return logger


def get_logger(name: str = "oscnet") -> logging.Logger:
    """
    Get an existing logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name) 