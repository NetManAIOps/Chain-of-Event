from flask import Flask
import logging
import logging.config
from os.path import abspath, dirname
import os
import json

logging.basicConfig()

if os.getenv('CURRENT_ENV') == 'DEV':
    logging_config_path = "/../conf/logging_dev.json"
else:
    logging_config_path = "/../conf/logging.json"

config_file = open(abspath(dirname(__file__) + logging_config_path))
LOG_CONFIG = json.loads(config_file.read())
logging.raiseExceptions = False
logging.config.dictConfig(LOG_CONFIG)
# logging.config.fileConfig(abspath(dirname(__file__)+"/../conf/logging.conf"))


def getLogger(name='unknown'):

    logger = logging.getLogger(name)

    return logger
