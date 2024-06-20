import numpy as np
import pandas as pd

emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
csv_file_path = '/root/autodl-tmp/exp_mmsa/datasets/MOSEI/mosei_all_label.csv'

if __name__== "__main__":

    df = pd.read_csv(csv_file_path)
    values = df[emotions].values.tolist()
    values = sum(values, [])
    values = list(set(values))
    values = sorted(values)
    # print(values) 
    # [0.0, 0.16666667, 0.33333334, 0.5, 0.6666667, 0.8333333, 1.0, 
    #  1.1666666, 1.3333334, 1.5, 1.6666666, 2.0, 
    #  2.1666667, 2.3333333, 2.6666667, 3.0]

    

