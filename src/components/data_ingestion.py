import os  # Module to interact with the operating system, like file and directory handling
import sys  # Provides system-specific parameters and functions
from src.exception import CustomException  # Custom module to handle exceptions
from src.logger import logging  # Custom module for logging

import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from dataclasses import dataclass  # To define classes with built-in methods like __init__ for creating and managing data attributes

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Define a configuration class to store file paths for data ingestion
@dataclass
class DataIngestionConfig:
    # Path to save the training data
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # Path to save the testing data
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # Path to save the raw dataset
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Main class for the data ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the configuration object containing file paths
        self.ingestion_config = DataIngestionConfig()

    # Method to handle the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Enter the dataIngestion Method or Component")  # Log entry message
        try:
            # Read the dataset from a CSV file
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            # Create the directory for storing data files if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")

            # Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to the specified path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing set to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the file paths for the train and test datasets
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            # Handle any exceptions by raising a CustomException
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    # Create an object of the DataIngestion class
    obj = DataIngestion()
    # Call the data ingestion method to execute the process
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    