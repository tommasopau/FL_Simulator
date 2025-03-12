import logging

def setup_logger(level: int = logging.INFO) -> None:
    """
    Configures the root logger with a console handler and formatter.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )