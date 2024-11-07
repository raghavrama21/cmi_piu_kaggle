# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.metrics import cohen_kappa_score

from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
pd.set_option("display.max_columns", 500)

train = pd.read_csv(r"C:\Users\ragha\Desktop\Competition\cmi_piu_kaggle\train.csv")

test = pd.read_csv(r"C:\Users\ragha\Desktop\Competition\cmi_piu_kaggle\test.csv")

data_dict = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\data_dictionary.csv"
)

train_ts = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\cmi_piu_kaggle\train_ts.csv"
)

test_ts = pd.read_csv(r"C:\Users\ragha\Desktop\Competition\cmi_piu_kaggle\test_ts.csv")

# %%
### There columns that are missing in the test
print(train.shape)
print(test.shape)

# %%
# %%
# Global configuration for PCIAT columns
PCIAT_COLS = [f"PCIAT-PCIAT_{i+1:02d}" for i in range(20)]
SII_CATEGORIES = ["0 (None)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"]


def recalculate_sii_vectorized(df):
    """
    Vectorized function to calculate the Severity Impairment Index (SII) based on PCIAT_Total score.

    Args:
        df (pd.DataFrame): DataFrame containing 'PCIAT-PCIAT_Total' and PCIAT item columns.

    Returns:
        pd.Series: Computed SII values.
    """
    total = df["PCIAT-PCIAT_Total"]
    max_possible = total + df[PCIAT_COLS].isna().sum(axis=1) * 5

    conditions = [
        (total <= 30) & (max_possible <= 30),
        (total.between(31, 49)) & (max_possible <= 49),
        (total.between(50, 79)) & (max_possible <= 79),
        (total >= 80) & (max_possible >= 80),
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=np.nan)


def add_sii_columns(df):
    """
    Adds SII-related columns to the DataFrame.

    Args:
        df (pd.DataFrame): The main dataset with 'PCIAT-PCIAT_Total' column.

    Returns:
        pd.DataFrame: Updated DataFrame with SII columns added.
    """
    df["recalc_sii"] = recalculate_sii_vectorized(df)

    # Map recalc_sii to readable categories
    sii_map = {0: "0 (None)", 1: "1 (Mild)", 2: "2 (Moderate)", 3: "3 (Severe)"}
    df["sii"] = df["recalc_sii"].map(sii_map).fillna("Missing")

    # Set ordered categories for consistency
    df["sii"] = pd.Categorical(df["sii"], categories=SII_CATEGORIES, ordered=True)

    # Mark complete responses
    df["complete_resp_total"] = df["PCIAT-PCIAT_Total"].where(
        df[PCIAT_COLS].notna().all(axis=1), np.nan
    )

    return df


def plot_sii_distribution(df):
    """
    Generates a bar plot showing the distribution of the SII.

    Args:
        df (pd.DataFrame): DataFrame with 'sii' column.

    Returns:
        None
    """
    sii_counts = (
        df[df["sii"] != "Missing"]["sii"]
        .value_counts()
        .reindex(SII_CATEGORIES)
        .reset_index()
    )
    sii_counts.columns = ["sii", "count"]
    total = sii_counts["count"].sum()
    sii_counts["percentage"] = (sii_counts["count"] / total) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x="sii", y="count", data=sii_counts, palette="Blues_d", ax=ax)
    ax.set_title("Distribution of Severity Impairment Index (SII)", fontsize=14)
    ax.set_xlabel("SII", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # Add count and percentage labels to each bar
    for i, (count, perc) in enumerate(
        zip(sii_counts["count"], sii_counts["percentage"])
    ):
        ax.text(i, count + 5, f"{int(count)} ({perc:.1f}%)", ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_pciat_total_distribution(df):
    """
    Generates a histogram showing the distribution of PCIAT_Total scores for complete responses.

    Args:
        df (pd.DataFrame): DataFrame with 'complete_resp_total' column.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["complete_resp_total"].dropna(), bins=20, ax=ax)
    ax.set_title("Distribution of PCIAT_Total for Complete Responses", fontsize=14)
    ax.set_xlabel("PCIAT_Total for Complete PCIAT Responses", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    plt.show()


def main(train_df):
    """
    Main function to process data and generate plots.

    Args:
        train_df (pd.DataFrame): Original training DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame with SII columns.
    """
    # Step 1: Calculate and add SII columns
    train_df = add_sii_columns(train_df)

    # Step 2: Plot SII distribution
    plot_sii_distribution(train_df)

    # Step 3: Plot PCIAT_Total distribution for complete responses
    plot_pciat_total_distribution(train_df)

    return train_df


# %%
processed_train = main(train)

# %%
train[train["sii"] != "Missing"].groupby(["complete_resp_total"])[
    "sii"
].count().reset_index()


# %%
def process_health_data(df):
    """
    Processes health data by performing a series of calculations and transformations,
    including physical health indexing, endurance scoring, normalization, and more.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing health-related features.

    Returns:
    pd.DataFrame: The DataFrame with additional processed columns.
    """
    try:
        # 1. Physical Health Index
        df["Physical_Health_Index"] = (
            df["Physical-BMI"]
            + df["Physical-Waist_Circumference"]
            + df["Physical-Diastolic_BP"]
            + df["Physical-Systolic_BP"]
        ) / 4
        logger.info("Calculated Physical_Health_Index.")

        # 2. Fitness Endurance Score
        df["Fitness_Endurance_Score"] = df["Fitness_Endurance-Max_Stage"] + (
            df["Fitness_Endurance-Time_Mins"] * 60 + df["Fitness_Endurance-Time_Sec"]
        )
        logger.info("Calculated Fitness_Endurance_Score.")

        # 3. Normalize Height and Weight
        for feature in ["Physical-Height", "Physical-Weight"]:
            if feature not in df.columns:
                logger.error(f"Feature '{feature}' not found in DataFrame columns.")
                raise ValueError(f"Feature '{feature}' not found in DataFrame.")

        scaler = StandardScaler()
        df[["Height_Norm", "Weight_Norm"]] = scaler.fit_transform(
            df[["Physical-Height", "Physical-Weight"]]
        )
        logger.info("Normalized Height and Weight.")

        # 4. Body Composition Ratios
        df["Fat_to_Lean_Mass_Ratio"] = df["BIA-BIA_Fat"] / df["BIA-BIA_FFM"]
        logger.info("Calculated Fat_to_Lean_Mass_Ratio.")

        # 5. Create Engagement Metric (weekly hours from daily)
        df["Computer_Engagement"] = df["PreInt_EduHx-computerinternet_hoursday"] * 7
        logger.info("Calculated Computer_Engagement (weekly hours).")

        # 6. Cumulative Scores
        df["Cumulative_PAQ"] = df["PAQ_A-PAQ_A_Total"] + df["PAQ_C-PAQ_C_Total"]
        logger.info("Calculated Cumulative_PAQ.")

        # 7. Analyze Zone Percentage (example for FGC)
        fgc_columns = [
            "FGC-FGC_CU",
            "FGC-FGC_CU_Zone",
            "FGC-FGC_GSND",
            "FGC-FGC_GSND_Zone",
            "FGC-FGC_GSD",
            "FGC-FGC_GSD_Zone",
        ]

        # Check that all required FGC columns are present
        missing_fgc_columns = [col for col in fgc_columns if col not in df.columns]
        if missing_fgc_columns:
            logger.error(f"Missing FGC columns: {missing_fgc_columns}")
            raise ValueError(f"Missing FGC columns: {missing_fgc_columns}")

        df["FGC_Percentage"] = df[fgc_columns].sum(axis=1) / len(fgc_columns)
        logger.info("Calculated FGC_Percentage.")

        # 8. Flag Missing Values for Core Health Data
        df["Missing_Health_Data"] = (
            df[["Physical-BMI", "Physical-Height", "Physical-Weight"]]
            .isnull()
            .any(axis=1)
            .astype(int)
        )
        logger.info("Flagged rows with Missing Health Data.")

    except Exception as e:
        logger.error("An error occurred during data processing.", exc_info=True)
        raise e

    return df


# Example usage:
train = process_health_data(train)
test = process_health_data(test)

# %%
train = pd.merge(train, train_ts, how="left", on="id")
test = pd.merge(test, test_ts, how="left", on="id")


# %%
def find_unique_columns(df1, df2):
    """
    Finds columns that are unique to each DataFrame.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    tuple: Two sets containing columns unique to df1 and df2, respectively.
    """
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)

    # Columns unique to each DataFrame
    unique_to_df1 = columns_df1 - columns_df2
    unique_to_df2 = columns_df2 - columns_df1

    return unique_to_df1, unique_to_df2


# Example usage
unique_columns_df1, unique_columns_df2 = find_unique_columns(train, test)
print("Columns unique to the first dataset:", unique_columns_df1)
print("Columns unique to the second dataset:", unique_columns_df2)

# %%
unique_columns_df1 = list(unique_columns_df1)
unique_columns_df1 = [
    item
    for item in unique_columns_df1
    if item not in ["sii", "recalc_sii", "complete_resp_total"]
]

# %%
train = train.drop(columns=unique_columns_df1)

# %%
train.shape

# %%
test.shape

# %%
train

# %%
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


def prepare_and_encode_categorical(train, test):
    # Fill missing values with "Missing" and convert to categorical in both train and test datasets
    for df in [train, test]:
        for c in cat_c:
            df[c] = df[c].fillna("Missing").astype("category")

    # Function to create a mapping dictionary where "Missing" is assigned 0
    def create_mapping(column, dataset):
        unique_values = dataset[column].unique()
        mapping = {
            value: idx + 1
            for idx, value in enumerate(unique_values)
            if value != "Missing"
        }
        mapping["Missing"] = 0  # Assign 0 to "Missing"
        return mapping

    # Encode categorical columns with integer mappings for both train and test
    for col in cat_c:
        mappingTrain = create_mapping(col, train)
        mappingTest = create_mapping(col, test)

        train[col] = train[col].replace(mappingTrain).astype(int)
        test[col] = test[col].replace(mappingTest).astype(int)

    return train, test


# Apply the function to the train and test datasets
train, test = prepare_and_encode_categorical(train, test)

# %%
train
