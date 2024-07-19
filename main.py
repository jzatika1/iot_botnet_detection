import argparse
import yaml
import traceback
from utils.logger import Logger
from data.loader import get_data_loader
from preprocessing import preprocess_data
from training import train_models
from classification import classify_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="IoT Botnet Detection System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--classify", action="store_true", help="Classify using trained models")
    parser.add_argument("--task", choices=['xgboost', 'lstm_cnn', 'both'], default='both', 
                        help="Select the task to run: xgboost, lstm_cnn, or both")
    return parser.parse_args()

def main():
    logger = Logger.setup('main')
    
    try:
        args = parse_arguments()
        
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        Logger.set_global_debug_mode(args.debug)
        
        logger.info("Starting IoT Botnet Detection System")
        
        data_loader = get_data_loader(config, logger=logger)
        
        if args.preprocess:
            preprocess_data(config, logger, args.task, data_loader)
        if args.train:
            train_models(config, logger, args.task)
        if args.classify:
            classify_data(config, logger, args.task)
        
        logger.info("IoT Botnet Detection System completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("IoT Botnet Detection System execution finished")

if __name__ == "__main__":
    main()