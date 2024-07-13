import logging
from pathlib import Path

class Logger:
    _debug_mode = False

    @classmethod
    def set_global_debug_mode(cls, debug):
        cls._debug_mode = debug

    @classmethod
    def setup(cls, name, debug=None):
        """
        Set up a logger with the specified name and debug mode.
        
        Args:
            name: The name of the logger (e.g., 'train' or 'classify').
            debug (optional): If provided, overrides the global debug mode for this logger.
        
        Returns:
            logging.Logger: Configured logger instance.
        """
        # Create logs directory if it doesn't exist
        log_dir = Path('logging/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # Use global debug mode if not specified
        debug = cls._debug_mode if debug is None else debug

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.handlers = []  # Clear any existing handlers

        # Create file handler
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Logging initialized for {name}. Debug mode: {debug}")
        if debug:
            logger.debug("Debug logging is enabled.")

        return logger

    @staticmethod
    def debug(logger, message):
        """
        Log a debug message.
        
        Args:
            logger: The logger instance.
            message: The debug message to log.
        """
        logger.debug(message)