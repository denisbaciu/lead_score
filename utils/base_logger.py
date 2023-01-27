import logging
import os
import sys

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    format='%(levelname)s [%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/logs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)