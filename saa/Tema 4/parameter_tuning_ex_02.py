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
# # 📝 Exercise M3.01
#
# The goal is to write an exhaustive search to find the best parameters
# combination maximizing the model generalization performance.
#
# Here we use a small subset of the Adult Census dataset to make the code
# faster to execute. Once your code works on the small subset, try to
# change `train_size` to a larger value (e.g. 0.8 for 80% instead of
# 20%).

# %%
import pandas as pd

from sklearn.model_selection import train_test_split

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer(
    [('cat_preprocessor', categorical_preprocessor,
      selector(dtype_include=object))],
    remainder='passthrough', sparse_threshold=0)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

# %% [markdown]
#
# Use the previously defined model (called `model`) and using two nested `for`
# loops, make a search of the best combinations of the `learning_rate` and
# `max_leaf_nodes` parameters. In this regard, you will need to train and test
# the model by setting the parameters. The evaluation of the model should be
# performed using `cross_val_score` on the training set. We will use the
# following parameters search:
# - `learning_rate` for the values 0.01, 0.1, 1 and 10. This parameter controls
#   the ability of a new tree to correct the error of the previous sequence of
#   trees
# - `max_leaf_nodes` for the values 3, 10, 30. This parameter controls the
#   depth of each tree.

# %%
# Write your code here.
from sklearn.model_selection import cross_val_score

learning_rate = [0.01, 0.1, 1, 10]
max_leaf_nodes = [3, 10, 30]

score = 0
params=[]

for x in learning_rate:
    for y in max_leaf_nodes:
        print("Probando modelo con learning rate=",x,"max leaf nodes",y)
        model.set_params(
            classifier__learning_rate=x,
            classifier__max_leaf_nodes=y
        )
        scores = cross_val_score(model, data_train, target_train, cv=2)
        mean_score = scores.mean()
        print("score:", mean_score)

        if mean_score > score:
            score = mean_score
            params = {'learning-rate': x, 'max leaf nodes': y}
print()
print("Mejor resultado final",score)
print("Mejores parametros",params)

# %% [markdown]
#
# Now use the test set to score the model using the best parameters
# that we found using cross-validation in the training set.

# %%
# Write your code here.
best_lr = params['learning-rate']
best_mln = params['max leaf nodes']

model.set_params(classifier__learning_rate=best_lr,
                 classifier__max_leaf_nodes=best_mln)
model.fit(data_train, target_train)
test_score = model.score(data_test, target_test)

print(f"Test score after the parameter tuning: {test_score:.3f}")
