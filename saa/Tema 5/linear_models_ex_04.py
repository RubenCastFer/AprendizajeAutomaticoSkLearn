# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise M4.04
#
# In the previous notebook, we saw the effect of applying some regularization
# on the coefficient of a linear model.
#
# In this exercise, we will study the advantage of using some regularization
# when dealing with correlated features.
#
# We will first create a regression dataset. This dataset will contain 2,000
# samples and 5 features from which only 2 features will be informative.

# %%
from sklearn.datasets import make_regression

data, target, coef = make_regression(
    n_samples=2_000,
    n_features=5,
    n_informative=2,
    shuffle=False,
    coef=True,
    random_state=0,
    noise=30,
)

# %% [markdown]
# When creating the dataset, `make_regression` returns the true coefficient
# used to generate the dataset. Let's plot this information.

# %%
import pandas as pd

feature_names = [
    "Relevant feature #0",
    "Relevant feature #1",
    "Noisy feature #0",
    "Noisy feature #1",
    "Noisy feature #2",
]
coef = pd.Series(coef, index=feature_names)
coef.plot.barh()
coef

# %% [markdown]
# Create a `LinearRegression` regressor and fit on the entire dataset and
# check the value of the coefficients. Are the coefficients of the linear
# regressor close to the coefficients used to generate the dataset?

# %%
# Write your code here.
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(data, target)
linear_regression.coef_

coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()
# %% [markdown]
# Now, create a new dataset that will be the same as `data` with 4 additional
# columns that will repeat twice features 0 and 1. This procedure will create
# perfectly correlated features.

# %%
# Write your code here.
import numpy as np

data = np.concatenate([data, data[:, [0, 1]], data[:, [0, 1]]], axis=1)

# %% [markdown]
# Fit again the linear regressor on this new dataset and check the
# coefficients. What do you observe?

# %%
# Write your code here.
linear_regression = LinearRegression()
linear_regression.fit(data, target)
linear_regression.coef_

feature_names = [
    "Relevant feature #0",
    "Relevant feature #1",
    "Noisy feature #0",
    "Noisy feature #1",
    "Noisy feature #2",
    "First repetition of feature #0",
    "First repetition of  feature #1",
    "Second repetition of  feature #0",
    "Second repetition of  feature #1",
]
coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()

# %% [markdown]
# Create a ridge regressor and fit on the same dataset. Check the coefficients.
# What do you observe?

# %%
# Write your code here.
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(data, target)
ridge.coef_

coef = pd.Series(ridge.coef_, index=feature_names)
_ = coef.plot.barh()
# %% [markdown]
# Can you find the relationship between the ridge coefficients and the original
# coefficients?

# %%
# Write your code here.
ridge.coef_[:5] * 3

import pandas as pd
from sklearn.model_selection import train_test_split

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values='?')
ames_housing = ames_housing.drop(columns="Id")

categorical_columns = ["Street", "Foundation", "CentralAir", "PavedDrive"]
target_name = "SalePrice"
X, y = ames_housing[categorical_columns], ames_housing[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_train["CentralAir"]


from sklearn.preprocessing import OneHotEncoder

single_feature = ["CentralAir"]
encoder = OneHotEncoder(sparse=False, dtype=np.int32)
X_trans = encoder.fit_transform(X_train[single_feature])
X_trans = pd.DataFrame(
    X_trans,
    columns=encoder.get_feature_names_out(input_features=single_feature),
)
X_trans


encoder = OneHotEncoder(drop="first", sparse=False, dtype=np.int32)
X_trans = encoder.fit_transform(X_train[single_feature])
X_trans = pd.DataFrame(
    X_trans,
    columns=encoder.get_feature_names_out(input_features=single_feature),
)
X_trans


from sklearn.pipeline import make_pipeline

model = make_pipeline(OneHotEncoder(drop="first", dtype=np.int32), Ridge())
model.fit(X_train, y_train)
n_categories = [X_train[col].nunique() for col in X_train.columns]
print(
    f"R2 score on the testing set: {model.score(X_test, y_test):.2f}"
)
print(
    f"Our model contains {model[-1].coef_.size} features while "
    f"{sum(n_categories)} categories are originally available."
)

