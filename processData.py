import csv
import glob

import numpy as np
import pandas as pd


# Combines all data files in a folder and filters as necessary. Saves into two new files
# pathToOutputFiles - where all the data files are located
# specific_files - file name filter to only combine the files that are wanted
# measure_filter - data filter that filters the values within each data file based. This is directly passed into `pd.query()`
def combineAndFilter(
    pathToOutputFiles="/",
    specific_files="*.csv",
    measure_filter=f'measure == "years_to_1_mfp" | measure == "rounds_to_1_mfp" | measure == "rounds_to_90_under_1_mfp" | measure == "years_to_90_under_1_mfp" | measure == "year_of_1_mfp_avg"',
):
    rows = []
    columns = []
    for filename in glob.glob(
        pathToOutputFiles + "**/" + specific_files, recursive=True
    ):
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            if len(columns) == 0:
                columns = next(reader)
            else:
                next(reader)
            rows.extend(reader)

    outputData = pd.DataFrame(rows, columns=columns)
    outputData.to_csv("combined_data.csv")

    filteredOutput = outputData.query(measure_filter)
    filteredOutput.to_csv("combined_filtered_data.csv")


combineAndFilter(
    pathToOutputFiles="C:/Users/adira/Documents/Github Repos/P_EPIONCHO-IBM/test_outputs/",
    specific_files="*-all_age_data.csv",
    measure_filter=f'measure == "years_to_1_mfp" | measure == "rounds_to_1_mfp" | measure == "rounds_to_90_under_1_mfp" | measure == "years_to_90_under_1_mfp" | measure == "year_of_1_mfp_avg"',
)
