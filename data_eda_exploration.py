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
# pl.set_option("display.max_columns", 500)

train = pl.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\train.csv"
)
test = pl.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\test.csv"
)

data_dictionary = pl.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\data_dictionary.csv"
)

# %%
# train.filter(pl.col("id").is_in(["00008ff9"]))

# # %%
# for i in list(data_dictionary.select("Instrument").unique()):
#     print(i)

# # %%
# data_dictionary.filter(pl.col("Instrument") == "Parent-Child Internet Addiction Test")

# # %%
# print("The number of participants in the training set: ", train.select("id").n_unique())
# print("The number of participants in the test set: ", test.select("id").n_unique())

# # %%
# # Define the root directory where the folders are located
# root_dir = r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\series_train.parquet"

# # Initialize an empty list to store each DataFrame
# dfs = []

# # Walk through each folder and find parquet files
# for foldername, subfolders, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.endswith(".parquet"):
#             file_path = os.path.join(foldername, filename)

#             # Read the parquet file into a Polars DataFrame
#             df = pl.read_parquet(file_path)

#             # Add a new column with the file name (without extension)
#             df = df.with_columns(pl.lit(foldername.split("\\")[-1]).alias("file_name"))

#             # Append the DataFrame to the list
#             dfs.append(df)

# # Concatenate all the DataFrames into one
# final_df = pl.concat(dfs)
# final_df.write_parquet(
#     r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\child-mind-institute-problematic-internet-use\collated_file.parquet"
# )
# %%
# final_df_lazy = pl.scan_parquet("collated_file.parquet")

# # %%
# print(final_df_lazy.columns)

# # %%
# final_df_lazy = final_df_lazy.filter((pl.col("non-wear_flag") == 0))

# # %%
# final_df_lazy = final_df_lazy.with_columns(
#     ((pl.col("X") ** 2 + pl.col("Y") ** 2 + pl.col("Z") ** 2).sqrt()).alias("Magnitude")
# )

# # %%
# final_df_lazy = final_df_lazy.drop(["X", "Y", "Z", "enmo"])

# # %%
# final_df_lazy = final_df_lazy.group_by("file_name").agg(
#     pl.mean("Magnitude").alias("Mean_Magnitude"),
#     pl.mean("light").alias("Mean_light"),
#     pl.mean("battery_voltage").alias("Mean_battery_voltage"),
# )

# # %%
# final_df_lazy = final_df_lazy.collect()

# # %%
# final_df_lazy.write_parquet(
#     r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\child-mind-institute-problematic-internet-use\aggregated_training_data.parquet"
# )

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
train.groupby(["sii"]).agg(
    {"id": "count", "PCIAT-PCIAT_Total": ["min", "max", "mean"]}
).reset_index()


# %%
## SII is incorrect in some cases
## refer to :https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/536407#3000620
PCIAT_cols = [f"PCIAT-PCIAT_{i+1:02d}" for i in range(20)]


def recalculate_sii(row):
    if pd.isna(row["PCIAT-PCIAT_Total"]):
        return np.nan
    max_possible = row["PCIAT-PCIAT_Total"] + row[PCIAT_cols].isna().sum() * 5
    if row["PCIAT-PCIAT_Total"] <= 30 and max_possible <= 30:
        return 0
    elif 31 <= row["PCIAT-PCIAT_Total"] <= 49 and max_possible <= 49:
        return 1
    elif 50 <= row["PCIAT-PCIAT_Total"] <= 79 and max_possible <= 79:
        return 2
    elif row["PCIAT-PCIAT_Total"] >= 80 and max_possible >= 80:
        return 3
    return np.nan


train["recalc_sii"] = train.apply(recalculate_sii, axis=1)

# %%
train.groupby(["recalc_sii"]).agg(
    {"id": "count", "PCIAT-PCIAT_Total": ["min", "max", "mean"]}
).reset_index()

# %%
train["sii"] = train["recalc_sii"]
train["complete_resp_total"] = train["PCIAT-PCIAT_Total"].where(
    train[PCIAT_cols].notna().all(axis=1), np.nan
)

sii_map = {0: "0 (None)", 1: "1 (Mild)", 2: "2 (Moderate)", 3: "3 (Severe)"}
train["sii"] = train["sii"].map(sii_map).fillna("Missing")

sii_order = ["0 (None)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"]
train["sii"] = pd.Categorical(train["sii"], categories=sii_order, ordered=True)

sii_counts = train[train["sii"] != "Missing"]["sii"].value_counts().reset_index()
sii_counts.columns = ["sii", "count"]

# Calculate total and percentages
total = sii_counts["count"].sum()
sii_counts["percentage"] = (sii_counts["count"] / total) * 100


# First plot: Bar plot for SII distribution
fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.barplot(x="sii", y="count", data=sii_counts, palette="Blues_d", ax=ax1)
ax1.set_title("Distribution of Severity Impairment Index (sii)", fontsize=14)
ax1.set_xlabel("SII", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)

# Add count and percentage labels to each bar
for p in ax1.patches:
    height = p.get_height()
    percentage = sii_counts.loc[sii_counts["count"] == height, "percentage"].values[0]
    ax1.text(
        p.get_x() + p.get_width() / 2,
        height + 5,
        f"{int(height)} ({percentage:.1f}%)",
        ha="center",
        fontsize=12,
    )

plt.tight_layout()

# Second plot: Histogram for PCIAT_Total distribution
fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.histplot(
    train[train["sii"] != "Missing"]["complete_resp_total"].dropna(), bins=20, ax=ax2
)
ax2.set_title("Distribution of PCIAT_Total for Complete Responses", fontsize=14)
ax2.set_xlabel("PCIAT_Total for Complete PCIAT Responses", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)

plt.tight_layout()

# Show both figures separately
plt.show()

# %$
## Looks like this alot of the responses are 0, lets look at the granular data

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


# %%
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

# %%
plot_distributions(train, "Basic_Demos-Sex")

#### Male participants have a higher SII score than women

# %%
#### Distribution by season
plot_distributions(train, "CGAS-Season")

#### Doesnt show much - can be excluded if we need to

# %%
train.head()

# %%
test.head()

# %%
train = train[train["sii"].notna()]
train["CGAS-CGAS_Score"].corr(train["complete_resp_total"])

# %%
#### Distribution by season
plot_distributions(train, "Physical-Season")

# %%
for i in [
    "Physical-BMI",
    "Physical-Height",
    "Physical-Weight",
    "Physical-Waist_Circumference",
    "Physical-Diastolic_BP",
    "Physical-HeartRate",
    "Physical-Systolic_BP",
]:
    print("Correlation of ", i, " response variable:")
    print(round(train[i].corr(train["complete_resp_total"]), 2) * 100)


# %%
### Lets look closer at the Height, Weight and Waist_Circumference
### since they had the highest correlation with the target
def average_distribution_by_buckets(df, variable, number_of_buckets):
    df[variable + "_bucket"] = pd.qcut(df[variable], q=number_of_buckets)
    print(df.groupby([variable + "_bucket"])["complete_resp_total"].mean())
    display(
        df.groupby([variable + "_bucket", "sii"])["id"]
        .count()
        .reset_index()
        .pivot_table(index=[variable + "_bucket"], columns="sii", values="id")
    )


average_distribution_by_buckets(train, "Physical-Height", 4)
average_distribution_by_buckets(train, "Physical-Weight", 4)
average_distribution_by_buckets(train, "Physical-Waist_Circumference", 4)

# %%
train[train["Physical-Weight"] >= 10].sort_values("Physical-Weight")
