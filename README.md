## GDP Growth Forecast


GPD Growth Forecast is a software that provides a command line interface tool
to train a model and make predictions about GDP Growth for the different countries in the world.

The way the software works is

- It processes the database data by removing columns with too many NaN and replacing the remaining with the mean of other samples, accoeding to different criterias (mean by year, mean by income group,       ...).
- The data is also organized in a convenient way for manipulation, with one column for each indicator.
- To reduce the number of features of the model, it is computed the shap values to select the most important features.
- Then, a gradient boosting regressor based model is trained with the selected features.
- Finally, the software allows to make predictions of GDP growth with the trained model.

Notes:

- Selected features and the model are stored in persistent files so that they can be used again.

#### Authors (NIU)

- Arturo del Cerro (1593930)
- Nil Bellmunt (1588664)
- Diego Palacios (1594283)

### Documentation

Take a look at the [documentation](documentation.html)!

## Usage

The `cli` script is the one to be executed by the user.
It is used to select the model features, train the model and make the predictions.

To select the features execute:
    
```bash
./cli.py select_features
```
       
You can also specify the number of features to select (default is 50). For example, to select 40 features execute:

```bash
./cli.py select_features -N 40
```

To train the model just run:

```bash
./cli.py train
```

To predict the GDP Growth for the 2011 year run:

```bash
./cli.py predict
```

The results are saved into the database. To predict a specified year (e.g. 1992) run

```bash
./cli.py predict --year 1992
```

The prediction year must be between 1961 and 2011.