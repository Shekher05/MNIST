# check the documentaion and work of logger 

import logging 
import os 
from datetime import datetime


# Create a logs directory if it doesn't exist 

LOG_File=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir=os.path.join(os.getcwd(), "logs")


os.makedirs(logs_dir, exist_ok=True)
# Set up the logging configuration
LOG_FILE_PATH=os.path.join(logs_dir,LOG_File)

# Ensure the log file path is correct
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

    )

