import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils.logger import Logger
import yaml
import numpy as np
from tqdm import tqdm

class DataLoader:
    def __init__(self, config, logger=None):
        self.config = config
        self.raw_dir = self.config['data']['raw_dir']
        self.batch_size = self.config['data']['batch_size']
        self.logger = logger or Logger.setup('DataLoader')

    def load_input_data(self):
        input_dir = self.config['data']['input_dir']
        self.logger.info(f"Loading input data from {input_dir}")
        
        all_data = []
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            data = self.load_file(file_path)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Loaded input data. Shape: {combined_data.shape}")
            return combined_data
        else:
            self.logger.warning("No input data loaded")
            return None

    def load_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                # Check if this is an UNSW-NB15 file
                if 'UNSW-NB15' in file_path:
                    column_names = [
                        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
                        'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
                        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                        'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
                        'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                        'attack_cat', 'Label'
                    ]
                    try:
                        data = pd.read_csv(file_path, names=column_names, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.logger.warning(f"UTF-8 decoding failed for {file_path}. Trying ISO-8859-1.")
                        data = pd.read_csv(file_path, names=column_names, encoding='ISO-8859-1')
                else:
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
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

    def remove_problematic_rows(self, X, threshold=1e15):
        initial_rows = X.shape[0]
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Remove rows with infinite values
        inf_mask = X[numeric_columns].isin([np.inf, -np.inf]).any(axis=1)
        X = X[~inf_mask]
        
        # Remove rows with extreme values
        extreme_mask = (X[numeric_columns].abs() > threshold).any(axis=1)
        X = X[~extreme_mask]
        
        removed_rows = initial_rows - X.shape[0]
        self.logger.info(f"Removed {removed_rows} rows with infinite or extreme values")
        self.logger.info(f"Rows removed due to infinity: {inf_mask.sum()}")
        self.logger.info(f"Rows removed due to extreme values: {extreme_mask.sum()}")
        return X

    def clean_dataset(self, features, labels):
        # Remove NaN values
        initial_rows = len(features)
        features_clean = features.dropna()
        labels_clean = labels.loc[features_clean.index]
        nan_removed = initial_rows - len(features_clean)
        
        # Remove problematic rows
        features_clean = self.remove_problematic_rows(features_clean)
        labels_clean = labels_clean.loc[features_clean.index]
        problematic_removed = len(labels_clean) - len(features_clean)
        
        total_removed = initial_rows - len(features_clean)
        self.logger.info(f"Removed {nan_removed} rows with NaN values and {problematic_removed} problematic rows")
        self.logger.info(f"Total rows removed: {total_removed}")
        self.logger.info(f"Remaining rows: {len(features_clean)}")
        
        return features_clean, labels_clean

    def load_and_combine_datasets(self):
        datasets = self.load_datasets()
        
        all_features = []
        all_labels = []
        
        for dataset_name, (data, _) in tqdm(datasets.items(), desc="Loading and Cleaning Datasets"):
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            features, labels = self.extract_features_and_labels(data, dataset_name)
            if features is None or labels is None:
                self.logger.info(f"Skipping dataset: {dataset_name}")
                continue
            
            # Clean the dataset
            features_clean, labels_clean = self.clean_dataset(features, labels)
            
            all_features.append(features_clean)
            all_labels.append(labels_clean)
        
        combined_features = pd.concat(all_features, axis=0, ignore_index=True)
        combined_labels = pd.concat(all_labels, axis=0, ignore_index=True)
        
        self.logger.info(f"Combined dataset shape: Features {combined_features.shape}, Labels {combined_labels.shape}")
        
        return combined_features, combined_labels

    def extract_features_and_labels(self, data, dataset_name):
        possible_label_columns = ['Label', 'label', 'type', 'Type', 'attack_cat']
        
        # Identify the label column
        label_column = next((col for col in possible_label_columns if col in data.columns), None)
        
        if label_column is None:
            self.logger.error(f"No suitable label column found in dataset '{dataset_name}'. Available columns: {data.columns.tolist()}")
            return None, None
        
        # Extract features and labels
        features = data.drop(columns=[col for col in possible_label_columns if col in data.columns]).copy()
        labels = data[label_column].copy()
        
        self.logger.info(f"Processing dataset: {dataset_name}")
        self.logger.info(f"Label column identified: {label_column}")
        
        # Dataset-specific processing
        if dataset_name.upper() == 'ROUTESMART':
            self.logger.info("ROUTESMART dataset detected. Treating all data as malicious.")
            labels = pd.Series(['malicious'] * len(data), index=data.index)
        elif dataset_name.upper() == 'UNSW-NB15':
            self.logger.info("UNSW-NB15 dataset detected. Using 'attack_cat' as label.")
            nan_mask = labels.isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                self.logger.info(f"Found {nan_count} NaN values in 'attack_cat'. Using 'Label' to categorize them.")
                labels = labels.mask(nan_mask & (data['Label'] == 0), 'benign')
                labels = labels.mask(nan_mask & (data['Label'] == 1), 'malicious')
        elif dataset_name.upper() == 'CICIDS17':
            self.logger.info("CICIDS17 dataset detected. Consolidating attack types.")
            labels = labels.replace({
                'Web Attack \x96 Brute Force': 'Web Attack',
                'Web Attack \x96 XSS': 'Web Attack',
                'Web Attack \x96 Sql Injection': 'Web Attack',
                'DoS slowloris': 'DoS',
                'DoS Slowhttptest': 'DoS',
                'DoS Hulk': 'DoS',
                'DoS GoldenEye': 'DoS'
            })
        
        # Common label processing
        labels = labels.fillna('Malicious (Unlabeled)')
        labels = labels.replace('', 'unknown')  # Replace empty strings with 'unknown'
        labels = labels.str.lower().str.replace(' ', '_')
        
        # Log information about labels and features
        self.logger.info(f"Unique labels: {labels.unique().tolist()}")
        self.logger.info(f"Number of unique labels: {len(labels.unique())}")
        self.logger.info(f"Label distribution:\n{labels.value_counts(dropna=False)}")
        self.logger.info(f"Features shape after extraction: {features.shape}")
        
        return features, labels

def get_data_loader(config, logger=None):
    return DataLoader(config, logger)
