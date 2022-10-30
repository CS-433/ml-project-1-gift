# Higgs Boson Challenge EPFL 2022 - Machine Learning (CS-433)
The repository contains all the material usedo to face the EPFL Machine Learning Higgs Boson Challenge (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) and it is organized as follows:
 - useful documentation and references consulted during the project (`documentation` folder)
 - (down-) loading of the dataset and (up-) loading of the results (`loading_dataset.py`)
 - preliminary inspection of the dataset (`feature_inspection.py`)
 - cleaning of the dataset (`cleaning_dataset.py`)
 - manipulation techniques on the features of the dataset (`feature_engineering.py`)
 - regression (`utilities_linear_regression.py`, `utilities_logistic_regression.py`, `implementations.py`)
 - final code to run, including the process that led to the final results (`run.py`)

# Team
This project is executed by the team GiFT with the following members:

Girolamo Vurro: [@girolamovurro14](https://github.com/girolamovurro14)

Francesca Venturi: [@francescaventurigit](https://github.com/francescaventurigit)

Tommaso Farnararo: @tommasofarnararo


# Project structure

## Presentation
The challenge aims at studying and  implementing manipulation techinques on the dataset, in order to perform datamining and, eventually, reliable predictions.

## Documentation
An effective and reliable data analysis is not able to leave a theoretical study of the problem out of consideration. A preliminary and, in our case, rough research on the Higgs Boson Challenge turns out to be fundamental. 

## Data Anlysis
Data inspection (`feature_inspection.py`) necessarily predates the actual data analysis. All the features are analyzed to catch the distribution, the skewness, the correlation, the symmetry around the mean, as well asto the standard statistical quantities.

## Data Mining
In addition to Logaritmic and Cosine tranforms, an optimal-degree polynomial expansion of the dataset is performed, too.

## Methods
All the regression algorithms studied so far during the course are implemented (`implementations.py`): Gradient Descent, Stochastic Gradient Descent, Least Squares, Ridge Regression, Logistic Regression with Gradient Descent (penalized and non-penalized), Logistic Regression with Stochastic Gradient Descent (penalized and non-penalized), Logistic Regression with Newton Method.

## Envirnoment
The project is developed and tested with `python3.8.10`. The required library for running the models and training is `numpy`. The library for visualization is `maptplotlib`. The Python IDE used is `Spyder`.

## Results
Results to predict the test datasets are generated by running the `run.py` script. The final results are saved in: `/data/finalsubmission.cvs`.
