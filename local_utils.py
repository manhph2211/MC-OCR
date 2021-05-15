import pandas as pd
import numpy as np
from ast import literal_eval


if __name__=='__main__':
    df = pd.read_csv('./data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv')
    annos = df['anno_polygons'].head(5).to_list()
    for anno in annos:
        anno = literal_eval(anno)
        for object in anno:
            category_id = object['category_id']
            print(category_id)



