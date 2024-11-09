# %%
import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    VotingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

SEED = 42
n_splits = 5

# %%
sample = pd.read_csv(r"sample_submission.csv")

train = pd.read_csv(r"train_final.csv")
train = train.dropna(subset="sii")

test = pd.read_csv(r"test_final.csv")

train = train.drop(["id", "sii", "complete_resp_total"], axis=1)
test = test.drop("id", axis=1)

cat_c = [
    "Basic_Demos-Enroll_Season",
    "CGAS-Season",
    "Physical-Season",
    "Fitness_Endurance-Season",
    "FGC-Season",
    "BIA-Season",
    "PAQ_A-Season",
    "PAQ_C-Season",
    "SDS-Season",
    "PreInt_EduHx-Season",
]


# %%
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(
        oof_non_rounded < thresholds[0],
        0,
        np.where(
            oof_non_rounded < thresholds[1],
            1,
            np.where(oof_non_rounded < thresholds[2], 2, 3),
        ),
    )


def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)


# %%
def TrainML(model_class, test_data):
    X = train.drop(["recalc_sii"], axis=1)
    y = train["recalc_sii"]

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    train_S = []
    test_S = []
    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)
    ):

        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(
            y_train, y_train_pred.round(0).astype(int)
        )
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test_data)

        print(
            f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}"
        )
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5],
        args=(y, oof_non_rounded),
        method="Nelder-Mead",
    )
    assert KappaOPtimizer.success, "Optimization did not converge."

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(
        f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}"
    )

    tpm = test_preds.mean(axis=1)
    tp_rounded = threshold_Rounder(tpm, KappaOPtimizer.x)

    return tp_rounded


# %%
# Model parameters for LightGBM
Params = {
    "learning_rate": 0.046,
    "max_depth": 12,
    "num_leaves": 478,
    "min_data_in_leaf": 13,
    "feature_fraction": 0.893,
    "bagging_fraction": 0.784,
    "bagging_freq": 4,
    "lambda_l1": 10,  # Increased from 6.59
    "lambda_l2": 0.01,  # Increased from 2.68e-06
}


# XGBoost parameters
XGB_Params = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1,  # Increased from 0.1
    "reg_lambda": 5,  # Increased from 1
    "random_state": SEED,
}


CatBoost_Params = {
    "learning_rate": 0.05,
    "depth": 6,
    "iterations": 200,
    "random_seed": SEED,
    "cat_features": cat_c,
    "verbose": 0,
    "l2_leaf_reg": 10,  # Increase this value
}

# Create model instances
Light = LGBMRegressor(**Params, random_state=SEED, verbose=-1, n_estimators=300)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

# Combine models using Voting Regressor
voting_model = VotingRegressor(
    estimators=[
        ("lightgbm", Light),
        ("xgboost", XGB_Model),
        ("catboost", CatBoost_Model),
    ]
)

# Train the ensemble model
Submission2 = TrainML(voting_model, test)

# %%
imputer = SimpleImputer(strategy="median")

ensemble = VotingRegressor(
    estimators=[
        (
            "lgb",
            Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("regressor", LGBMRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "xgb",
            Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("regressor", XGBRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("regressor", CatBoostRegressor(random_state=SEED, silent=True)),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("regressor", RandomForestRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "gb",
            Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("regressor", GradientBoostingRegressor(random_state=SEED)),
                ]
            ),
        ),
    ]
)

Submission3 = TrainML(ensemble, test)


# In[7]:


Submission3 = pd.DataFrame({"id": sample["id"], "sii": Submission3})
Submission2 = pd.DataFrame({"id": sample["id"], "sii": Submission2})


# In[8]:


sub2 = Submission2
sub3 = Submission3


sub2 = sub2.sort_values(by="id").reset_index(drop=True)
sub3 = sub3.sort_values(by="id").reset_index(drop=True)

combined = pd.DataFrame({"id": sub2["id"], "sii_2": sub2["sii"], "sii_3": sub3["sii"]})


def majority_vote(row):
    return row.mode()[0]


combined["final_sii"] = combined[["sii_2", "sii_3"]].apply(majority_vote, axis=1)

final_submission = combined[["id", "final_sii"]].rename(columns={"final_sii": "sii"})

final_submission.to_csv("submission.csv", index=False)

print("Majority voting completed and saved to 'Final_Submission.csv'")


# In[ ]:
