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

# %%
pd.set_option("display.max_columns", 500)

train = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\train.csv"
)
test = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\test.csv"
)

data_dict = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\data_dictionary.csv"
)

# %%
### There columns that are missing in the test
print(train.shape)
print(test.shape)

# %%
nan_percentage = train.isna().mean() * 100
nan_percentage = nan_percentage.reset_index().sort_values(0, ascending=False)
display(nan_percentage.T)
columns_to_remove = list(nan_percentage[nan_percentage[0] > 50]["index"])
columns_to_impute = list(
    nan_percentage[(nan_percentage[0] > 10) & (nan_percentage[0] < 50)]["index"]
)
columns_w_less_than_10_perc = list(nan_percentage[(nan_percentage[0] < 10)]["index"])

train_filtered = train[columns_w_less_than_10_perc + columns_to_impute]

# %%
train_filtered.select_dtypes(include="number").columns

# %%
## columns not in test
train_cols = set(train.columns)
test_cols = set(test.columns)
columns_not_in_test = sorted(list(train_cols - test_cols))
print(len(data_dict[data_dict["Field"].isin(columns_not_in_test)]))
data_dict[data_dict["Field"].isin(columns_not_in_test)][["Field", "Description"]]

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
print(round(train[train["complete_resp_total"] == 0]["Basic_Demos-Age"].median(), 2))
print(round(train[train["complete_resp_total"] != 0]["Basic_Demos-Age"].median(), 2))

train.groupby(["sii", "Basic_Demos-Age"])["id"].count().reset_index().pivot_table(
    columns="Basic_Demos-Age", index="sii", values="id"
)


# %%
def plot_distributions(data, variable):
    # Plot 1: Count Distribution of Basic_Demos-Age (Histogram)
    grouped_age = data.groupby([variable])["id"].count().reset_index()

    fig1 = px.histogram(
        grouped_age,
        x=variable,
        y="id",
        labels={
            variable: variable,
            "id": "Count",
        },
        title="Count Distribution of SII by " + variable,
    )

    fig1.update_layout(
        xaxis_title=variable,
        yaxis_title="Count",
        template="plotly_white",
        width=900,
        height=600,
    )

    fig1.show()

    # Plot 2: Count Distribution of SII by Basic_Demos-Age (Line Chart)
    grouped_sii_age = data.groupby(["sii", variable])["id"].count().reset_index()

    fig2 = px.line(
        grouped_sii_age,
        x=variable,
        y="id",  # Use counts of 'id'
        color="sii",  # Separate lines for each SII category
        markers=True,
        labels={
            variable: variable,
            "id": "Count",
            "sii": "Severity Impairment Index (SII)",
        },
        title="Count Distribution of SII by " + variable,
    )

    fig2.update_layout(
        xaxis_title=variable,
        yaxis_title="Count",
        legend_title="SII",
        template="plotly_white",
        width=900,
        height=600,
    )

    fig2.show()

    # Plot 3: Percentage Distribution of SII by Basic_Demos-Age (Line Chart)
    grouped_percentage = data.groupby(["sii", variable])["id"].count().reset_index()

    grouped_percentage["percentage"] = grouped_percentage.groupby(variable)[
        "id"
    ].transform(lambda x: (x / x.sum()) * 100)

    fig3 = px.line(
        grouped_percentage,
        x=variable,
        y="percentage",
        color="sii",  # Separate lines for each SII category
        markers=True,
        labels={
            variable: variable,
            "percentage": "Percentage (%)",
            "sii": "Severity Impairment Index (SII)",
        },
        title="Percentage Distribution of SII by " + variable,
    )

    fig3.update_layout(
        xaxis_title=variable,
        yaxis_title="Percentage (%)",
        legend_title="SII",
        template="plotly_white",
        width=900,
        height=600,
    )

    fig3.show()


# Usage
plot_distributions(train, "Basic_Demos-Age")

#### There is definitely a specific age group that has a proclivity for high SIIs
#### Between 13 - 18 Yrs as we can see the lines cross over the most during that age group

# %%
#### Lets look at the split of genders

print(
    train[train["complete_resp_total"] == 0].groupby(["Basic_Demos-Sex"])["id"].count()
)
print(
    train[train["complete_resp_total"] != 0].groupby(["Basic_Demos-Sex"])["id"].count()
)

plot_distributions(train, "Basic_Demos-Sex")

#### Male participants have a higher SII score than women

# %%
#### Distribution by season
plot_distributions(train, "CGAS-Season")


# %%
def calculate_physical_health_index(df):
    """
    Calculate the Physical Health Index for the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing physical health measurements.

    Returns:
        pd.Series: A Series containing the Physical Health Index for each individual.
    """
    # Ensure required columns are in the DataFrame
    required_columns = [
        "Physical-BMI",
        "Physical-Waist_Circumference",
        "Physical-Diastolic_BP",
        "Physical-Systolic_BP",
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("One or more required columns are missing from the DataFrame.")

    # Calculate the Physical Health Index
    physical_health_index = (
        df["Physical-BMI"]
        + df["Physical-Waist_Circumference"]
        + df["Physical-Diastolic_BP"]
        + df["Physical-Systolic_BP"]
    ) / len(required_columns)

    return physical_health_index


# %%


# %%


def print_correlations(df, response_variable):
    """
    Print the correlation of specified health metrics with the response variable.

    Args:
        df (pd.DataFrame): DataFrame containing health metrics and response variable.
        response_variable (str): The name of the response variable to correlate against.
    """
    health_metrics = [
        "Physical-BMI",
        "Physical-Height",
        "Physical-Weight",
        "Physical-Waist_Circumference",
        "Physical-Diastolic_BP",
        "Physical-HeartRate",
        "Physical-Systolic_BP",
        "Physical_Health_Index",
    ]

    for metric in health_metrics:
        if metric in df.columns:
            correlation = df[metric].corr(df[response_variable])
            print(
                f"Correlation of {metric} with {response_variable}: {round(correlation * 100, 2)}%"
            )
        else:
            print(f"Warning: {metric} is not in the DataFrame.")


if __name__ == "__main__":
    train["Physical_Health_Index"] = calculate_physical_health_index(train)
    print_correlations(train, "complete_resp_total")

# %%
train[[]]
