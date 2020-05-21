import subprocess
import pandas as pd
from LASExplanation.LIMEBAG import *
import os
def main():
    file = os.path.join("LASExplanation", 'camel-1.2.csv')
    df = pd.read_csv(file)

    # print(subprocess.Popen("echo pkw", shell=True, stdout=subprocess.PIPE).stdout.read())
    # ps = subprocess.Popen("type lime_fi.txt", shell=True,stdout=subprocess.PIPE)
    # print('Text copied.')
    # subprocess.Popen("sk.py --text 30 --latex True --higher True", shell=True,stdin=ps.stdout)
    # print('sk.py called')
    bag = LIMEBAG()
    return 0



if __name__ == "__main__":
    main()