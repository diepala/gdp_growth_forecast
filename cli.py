#!/usr/bin/env python

"""
CLI script.

This is the script to be executed by the user.
It is used to select the model features, train the model and make the predictions.

Usage:
    To select the features execute:
    
    .. code-block:: bash

        ./cli.py select_features
        
    You can also specify the number of features to select (default is 50). For example, to select 40 features execute:
    
    .. code-block:: bash

        ./cli.py select_features -N 40
        
    To train the model just run:
    
    .. code-block:: bash

        ./cli.py train
        
    To predict the GDP Growth for the 2011 year run:
    
    .. code-block:: bash

        ./cli.py predict
    
    The results are saved into the database. To predict a specified year (e.g. 1992) run
    
    .. code-block:: bash

        ./cli.py predict --year 1992
        
    The prediction year must be between 1961 and 2011.
"""

import os
import sys
import logging
import logging.config
import argparse
from datetime import datetime
import json

from utils import config, io, models, feature_selection

# load logger configuration from file
logging.config.fileConfig(config.LOG_CONFIG_PATH, defaults={'logfilename' : os.path.join(config.LOGS_PATH, datetime.now().strftime('cli_%Y-%m-%d_%H:%M:%S.log'))})
logger = logging.getLogger('main')

# parser
parser = argparse.ArgumentParser(description="CLI tool to train and predict GPD Growth of countries.")
parser.add_argument(
    "task",
    choices=["train", "predict", "select_features"],
    help="Task to be performed",
)
parser.add_argument('--year',
                    default=2011,
                    dest='year',
                    type=int,
                    help='Year to predict. Only for task=predict')
parser.add_argument('-N',
                    default=50,
                    dest='number_features',
                    type=int,
                    help='Number of features to select. Only for task=select_features')


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    if args.task == "select_features":
        
        logger.info("Selecting features")
        
        X, y = io.retrieve_training_dataset('all') # get dataset
        
        model = models.GDPGrowthPredictor() # untrained model to select the features
        
        features = feature_selection.select_features_with_shap(k=args.number_features, model=model.get_internal_model(), model_type='tree', X=X, y=y) # select features
        
        logger.info(f"Writting results to {config.FEATURES_PATH}")
        # save features to file
        with open(config.FEATURES_PATH, 'w') as f:
            json.dump(features, f)
            
    
    if args.task == "train":
        
        logger.info("Training")
        
        if not os.path.exists(config.FEATURES_PATH):
            logger.error("Features are not selected. Run 'cli.py select_features' first.")
            sys.exit()

        X, y = io.retrieve_training_dataset(config.FEATURES_PATH) # get dataset

        model = models.GDPGrowthPredictor() # build model
        
        model.train(X, y)
        
        logger.info(f"Saving model to {config.MODEL_FILE_PATH}")
        model.save(config.MODEL_FILE_PATH)


    if args.task == "predict":

        logger.info("Predinting")

        if not os.path.exists(config.MODEL_FILE_PATH):
            logger.error("The model is not trained. Run 'cli.py train' first.")
            sys.exit()

        if not os.path.exists(config.FEATURES_PATH):
            logger.error("Features are not selected. Run 'cli.py select_features' first.")
            sys.exit()

        year = args.year

        if year > 2011 or year < 1961:
            logger.error("Year must be between 1961 and 2011.")
            sys.exit()

        X, country_codes = io.retrieve_predict_dataset(config.FEATURES_PATH, year=year) # get dataset

        model = models.GDPGrowthPredictor.load(config.MODEL_FILE_PATH)

        y = model.predict(X)

        logger.info(f"GPD Growth for year {year} predicted")

        logger.info(f"Writting results database")
        io.write_predictions_to_database(country_codes, year, y)

