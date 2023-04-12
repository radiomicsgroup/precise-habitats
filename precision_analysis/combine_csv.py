### Concatenate csv files with repeatability/reproducibility data
import os
import glob
import pandas as pd
import sys

argv = sys.argv[1:]
os.chdir(argv[0]) 
filename = argv[1]
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv(filename, index=False, encoding='utf-8-sig')

## INPUT
# argv[0]: /nfs/rnas/PRECISION/COHORT/pipelines/analysis/csv_files #folder containing csv files
# argv[1]: /nfs/rnas/PRECISION/COHORT/pipelines/analysis/repro_df_cohort.csv #output major csv with all minor csv concatenated

## EXAMPLE:
#python combine_csv.py /nfs/rnas/PRECISION/COHORT/pipelines/analysis/csv_files /nfs/rnas/PRECISION/COHORT/pipelines/analysis/repro_df_cohort.csv