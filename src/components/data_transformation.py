import sys 
import os  
from dataclasses import dataclass 
import numpy as np  
import pandas as pd  
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

from src.exception import CustomException  
from src.logger import logging 
from src.utils import save_object  # Utility function to save Python objects to a file

# Define a configuration class to store the path for the preprocessor object file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # File path for saving the preprocessor object


# Main class for data transformation
class DataTransformation:
    def __init__(self):
        # Initialize the configuration attribute with an instance of DataTransformationConfig
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating and returning the data transformation pipeline.
        '''
        try:
            # Define the numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns: handle missing values and scale data
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
                    ("scaler", StandardScaler())  # Standardize the numerical data
                ]
            )

            # Pipeline for categorical columns: handle missing values, encode categories, and scale data
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Convert categories to one-hot encoded columns
                    ("scaler", StandardScaler(with_mean=False))  # Scale encoded categorical data
                ]
            )

            # Log column information for debugging
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the pipelines into a ColumnTransformer to apply them to respective columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor  # Return the preprocessor object

        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function applies data transformation on the train and test datasets and saves the preprocessor object.
        '''
        try:
            # Load the training and testing datasets from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply the transformations to the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target column for training and testing datasets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessor object to utils a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed training and testing data arrays and the path to the preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)
