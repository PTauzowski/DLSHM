import glob
import os.path

import pandas as pd
from openpyxl import load_workbook
from pandas import ExcelWriter


def prepare_excel_multiaugmented_results(TASK_PATH,MODEL_NAME,augmentations,nrows):
    writer = ExcelWriter(os.path.join(TASK_PATH,MODEL_NAME)+'_results.xlsx', engine='openpyxl')
    start_row=0
    start_col=0
    for name, postfix, fn in augmentations:

        dircontent=os.path.join(TASK_PATH,MODEL_NAME+postfix)
        excel_results = glob.glob(dircontent + '/ExcelResults/*')
        result=excel_results[0]
        df = pd.read_excel(result,skiprows=9, nrows=nrows, engine='openpyxl')
        df.at[10, 0] = name

        df.to_excel(writer, index=False, header=True if start_row == 0 else False,
                    startrow=start_row, startcol=start_col )

        if start_col == 0:
            start_col = 10
        else:
            start_col = 0
            start_row += nrows + 2

    writer.close()

