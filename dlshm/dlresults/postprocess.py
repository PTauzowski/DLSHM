import glob


def prepare_excel_multiaugmented_results(basepathname):
    files = glob.glob(basepathname + '*')
    for file in files: