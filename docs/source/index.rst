.. gdp_growth_forecast documentation master file, created by
   sphinx-quickstart on Mon Dec  7 13:15:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gdp_growth_forecast's documentation!
===============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   api/modules


GDP Growth Forecast
===================

GPD Growth Forecast is a software that provides a command line interface tool
to train a model and make predictions about GDP Growth for the different countries in the world.

The way the software works is:

    - It processes the database data by removing columns with too many NaN and replacing the remaining with the mean of other samples, accoeding to different criterias (mean by year, mean by income group,       ...).
    - The data is also organized in a convenient way for manipulation, with one column for each indicator.
    - To reduce the number of features of the model, it is computed the shap values to select the most important features.
    - Then, a gradient boosting regressor based model is trained with the selected features.
    - Finally, the software allows to make predictions of GDP growth with the trained model.
    
Notes:
    - Selected features and the model are stored in persistent files so that they can be used again.
    
Usage
=====

For how to use the script, see :py:mod:`cli`.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
