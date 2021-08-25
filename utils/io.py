"""
IO module for retrieving the dataset and loadding predictions to the database.

Contains different functions for retrieving tables from the database, retriving the processed data,
writting the predictions to the database, and retrieving the dataset for training and prediction.
"""

from typing import Union, List, Tuple
import logging
import json
import sqlite3
import pandas as pd
import numpy as np
from utils import config
import utils.preprocessing


def retrieve_table(name: str) -> pd.DataFrame:
    """
    Read a table from the database.

    Args:
        name (str): Name of the table.
            Possible values are ``'Countries'``, ``'CountryNotes'``, ``'Footnotes'``, ``'CountryIndicators'``,
            ``'Indicators'``, ``'IndicatorsNotes'`` and ``'EstimatedGPDGrowth'``

    Returns:
        DataFrame: The specified table.

    Raises:
        ValueError: If `name` is not a valid table.
    """

    valid_names = ['Countries', 'CountryNotes', 'Footnotes', 'CountryIndicators',
                   'Indicators', 'IndicatorsNotes', 'EstimatedGPDGrowth']

    if name not in valid_names:
        raise ValueError(f"'{name}' is not a valid table. Should be one of {valid_names}.")
    
    logging.getLogger(__name__).info(f"Reading database table '{name}'")
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        table = pd.read_sql_query(f"SELECT * from {name}", conn)

    return table

def write_predictions_to_database(country_codes: np.array, year: int, values: np.array):
    """
    Writes the predictions to the `EstimatedGPDGrowth` table in the database.
    
    Args:
        country_codes (array): Array with the country codes of the predictions.
        year (int): Year of the predictions.
        values (array): Array with the predicted values.
    """
    
    df = pd.DataFrame({'CountryCode': country_codes, 'Year': year, 'Value': values}) # dataframe with the data
    
    logging.getLogger(__name__).info(f"Writting predictions of year {year} to database")
    # write to database
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        df.to_sql('EstimatedGPDGrowth', conn, if_exists='replace', index=False)
    

def retrieve_processed_dataset() -> pd.DataFrame:
    """
    Retrieves the processed dataset.
    
    This function reads the data from the database, prepares the data by ordering and removing NaN values,
    encodes the categorical features into numbers and adds the target variable to predict.
    
    Returns:
        DataFrame: The processed dataset.
    """
    logger = logging.getLogger(__name__)
    
    # get the prepared data
    df_country_indicators = utils.io.retrieve_table('CountryIndicators')
    df_countries = data_countries = utils.io.retrieve_table('Countries')
    logger.info('Database tables retrieved')
    data = utils.preprocessing.prepare_data(df_country_indicators, df_countries)
    # encode categorical variables
    data = utils.preprocessing.encode_dataset(data)
    # add target variable to predict
    utils.preprocessing.add_target_variable(data)
    logger.info('Dataset processed')

    return data

def retrieve_training_dataset(features: Union[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves the training dataset, with the selected features.
    
    Args:
        features (str, List[str]): If 'all', all the numeric features are returned.
            It can also be a list with the name of the features to select,
            or the filename of a json file containing the list of features.
            
    Returns:
        X (pd.DataFrame): Training dataset features.
        y (pd.DataFrame): Target variable to predict.
    """

    logger = logging.getLogger(__name__)
    
    data = utils.io.retrieve_processed_dataset().select_dtypes(include='number').dropna() # drop rows without target variable
    logger.info('Processed dataset loaded')
    y = data['target']
    X = data.drop(columns='target')

    logger.info('Selecting dataset features')
    if features=='all':
        return X, y
    
    if isinstance(features, str):
        if not features.endswith('.json'):
            raise ValueError('file must be in .json format')
        with open(features) as f:
            features = json.load(f)
    
    X = X[features] # select features
    
    return X, y


def retrieve_predict_dataset(features: Union[str, List[str]], year: int) -> Tuple[pd.DataFrame, np.array]:
    """
    Retrieves the predict dataset, with the selected features.

    Selected features must have the same features the model have been trained with.
    
    Args:
        features (str, List[str]): If 'all', all the numeric features are returned.
            It can also be a list with the name of the features to select,
            or the filename of a json file containing the list of features.
        year (int): Year to predict. This will retrieve the necessary data to predict this year.
            
    Returns:
        X (pd.DataFrame): Predict dataset features.
        country_codes (np.array): Country codes corresponding to the dataset.
    """

    logger = logging.getLogger(__name__)
    
    data = utils.io.retrieve_processed_dataset()
    logger.info('Processed dataset loaded')
    data = data.loc[data['Year']==(year-1)] # select rows to predict for the specified year
    country_codes = data['CountryCode'].values # store the country codes, as they are dropped in the next line
    X = data.select_dtypes(include='number').drop(columns='target')
    logger.info('Samples to predict selected')

    logger.info('Selecting dataset features')
    if features=='all':
        return X, country_codes
    
    if isinstance(features, str):
        if not features.endswith('.json'):
            raise ValueError('file must be in .json format')
        with open(features) as f:
            features = json.load(f)
    
    X = X[features] # select features
    
    return X, country_codes
