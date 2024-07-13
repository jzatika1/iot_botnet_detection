import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils.logger import Logger
import yaml

class DataLoader:
    def __init__(self, config, logger=None):
        self.config = config
        self.raw_dir = self.config['data']['raw_dir']
        self.batch_size = self.config['data']['batch_size']
        self.logger = logger or Logger.setup('DataLoader')

    def load_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                try:
                    # First attempt with UTF-8
                    data = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with ISO-8859-1
                    self.logger.warning(f"UTF-8 decoding failed for {file_path}. Trying ISO-8859-1.")
                    data = pd.read_csv(file_path, encoding='ISO-8859-1')
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_path}")
                return None
            
            # Strip whitespace from column names
            data.columns = data.columns.str.strip()
            
            return data
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def process_dataset(self, dataset_name):
        dataset_config = self.config['data']['datasets'][dataset_name]
        if not dataset_config['enabled']:
            self.logger.info(f"Skipping disabled dataset: {dataset_name}")
            return None, None
    
        dataset_path = os.path.join(self.raw_dir, dataset_name)
        self.logger.info(f"Processing dataset: {dataset_name}")
    
        file_paths = [os.path.join(dataset_path, f) for f in dataset_config['files']]
        self.logger.info(f"Found {len(file_paths)} files in {dataset_name}")
    
        all_data = []
        file_data_lengths = {}
    
        with Pool(processes=cpu_count()) as pool:
            self.logger.info(f"Using a pool of {cpu_count()} processes")
            for i in range(0, len(file_paths), self.batch_size):
                batch_files = file_paths[i:i + self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch_files)} files")
                batch_data = pool.map(self.load_file, batch_files)
                for file, data in zip(batch_files, batch_data):
                    if data is not None:
                        all_data.append(data)
                        file_data_lengths[file] = len(data)
                        Logger.debug(self.logger, f"Processed file: {file}, loaded {len(data)} records")
                self.logger.info(f"Batch {i//self.batch_size + 1} processed. Total records so far: {sum(file_data_lengths.values())}")
    
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Processing completed for {dataset_name}. Total records: {len(combined_data)}")
            self.logger.info(f"Data shape: {combined_data.shape}")
            Logger.debug(self.logger, f"Columns: {combined_data.columns.tolist()}")
    
            return combined_data, file_data_lengths
        else:
            self.logger.warning(f"No data loaded for dataset: {dataset_name}")
            return None, None

    def load_datasets(self):
        datasets = {}
        for dataset_name in self.config['data']['datasets']:
            if self.config['data']['datasets'][dataset_name]['enabled']:
                data, file_lengths = self.process_dataset(dataset_name)
                if data is not None:
                    datasets[dataset_name] = (data, file_lengths)
        return datasets

    def load_test_datasets(self):
        # This method should be implemented to load test datasets
        # For now, we'll just return an empty dictionary
        self.logger.warning("load_test_datasets method not implemented yet")
        return {}

def get_data_loader(config, logger=None):
    return DataLoader(config, logger)
