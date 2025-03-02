"""
Logging utility for the NEAR Cortex-1 project.
Provides consistent logging across all modules.
"""

import logging
import os
import sys
from typing import Optional

def setup_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger. If None, use the root logger.
        level: Logging level to use.
        
    Returns:
        A configured logger instance.
    """
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
        
    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(level)
    return logger 