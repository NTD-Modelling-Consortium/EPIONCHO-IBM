import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_outputs(file_name):
    df = pd.read_csv("test_outputs/python_model_output/" + file_name + ".csv")
    for measure in df["measure"].unique():
        measure_columns = "_" + measure
        filtered_df = df[((df["measure"] == measure))]
        numbers = df[((df["measure"] == "number"))]

        filtered_df.fillna(0, inplace=True)
        numbers.fillna(0, inplace=True)

        if measure == "number":
            measure_columns = "_numbers_2"
        new_combined_df = pd.merge(
            filtered_df,
            numbers,
            on=["year_id", "age_start", "age_end"],
            suffixes=(measure_columns, "_number"),
        )
        for col in df.columns:
            if col.startswith("draw"):
                new_combined_df[col] = (
                    new_combined_df[col + measure_columns]
                    * new_combined_df[col + "_number"]
                )
        new_combined_df = new_combined_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in [measure_columns]
            ]
        )

        age_grouped_df = new_combined_df.copy()

        for col in df.columns:
            if col.startswith("draw"):
                age_grouped_df[col] = (
                    age_grouped_df[col] / age_grouped_df[col + "_number"]
                )
        age_grouped_df = age_grouped_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in ["_number"]
            ]
        )

        age_groups = [
            str(x) + "_" + str(y)
            for x, y in zip(age_grouped_df["age_start"], age_grouped_df["age_end"])
        ]
        years = age_grouped_df["year_id"].values
        measure_values = age_grouped_df.iloc[:, 5:]

        new_combined_df = (
            new_combined_df.drop(
                columns=[
                    "age_start",
                    "age_end",
                    "measure" + measure_columns,
                    "measure_number",
                ]
            )
            .groupby("year_id")
            .sum()
            .reset_index()
        )

        for col in df.columns:
            if col.startswith("draw"):
                new_combined_df[col] = (
                    new_combined_df[col] / new_combined_df[col + "_number"]
                )
        new_combined_df = new_combined_df.drop(
            columns=[
                col + suffix
                for col in df.columns
                if col.startswith("draw")
                for suffix in ["_number"]
            ]
        )

        years_2 = new_combined_df["year_id"]
        measure_values_2 = new_combined_df.iloc[:, 1:]
        age_groups_2 = np.full(len(years_2), "0_80")

        def create_graphs(newDf, newDf_2, print_num):
            index = 0
            save_file_name = file_name
            square = math.floor(math.sqrt(len(newDf["age_groups"].unique()) + 1))
            fig, ax1 = plt.subplots(
                square, square + 1, figsize=(15, 15), sharex=True, sharey=True
            )
            ax1 = ax1.flatten()
            for age_group in newDf["age_groups"].unique():
                ax = ax1[index]
                group_data = newDf[newDf["age_groups"] == age_group]
                ax.plot(group_data["years"], group_data["measure"], label=age_group)
                ax.vlines(
                    x=1988, color="red", ymin=0, ymax=np.max(group_data["measure"])
                )
                ax.set_xlim(left=1950)
                ax.set_title(age_group)
                if index % (square + 1) == 0:
                    ax.set_ylabel(ylab)
                index += 1
            ax1[index].plot(
                newDf_2["years"],
                newDf_2["measure"],
                label=newDf_2["age_groups"].unique(),
            )
            ax1[index].vlines(
                x=1988, color="red", ymin=0, ymax=np.max(newDf_2["measure"])
            )
            ax1[index].set_xlim(left=1950)
            ax1[index].set_title(newDf_2["age_groups"].unique())
            for i in range(index + 1, len(ax1)):
                ax1[i].tick_params(
                    left=False, labelleft=False, bottom=False, labelbottom=False
                )
            plt.savefig(
                "test_outputs/all_measure_graphs/"
                + measure
                + "_"
                + save_file_name
                + ".png",
                dpi=300,
            )
            plt.close(fig)

        newDf = pd.DataFrame(
            {
                "years": years,
                "measure": measure_values.mean(axis=1, skipna=True).tolist(),
                "age_groups": age_groups,
            }
        )

        newDf_2 = pd.DataFrame(
            {
                "years": years_2,
                "measure": measure_values_2.mean(axis=1, skipna=True).tolist(),
                "age_groups": age_groups_2,
            }
        )
        ylab = measure + " Prevalence"
        if measure == "number":
            ylab = "Population Count"
        create_graphs(newDf, newDf_2, print_num=False)
