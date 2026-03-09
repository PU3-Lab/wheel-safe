import glob
import os

import pandas as pd

from lib.utils.path import output_path


def combine_csv(input_path):
    # 1. 합치고 싶은 파일들이 있는 경로 설정
    all_files = glob.glob(os.path.join(input_path, 'slope_labels_*.csv'))

    # 2. 모든 CSV 파일을 불러와 리스트에 저장
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    # 3. 리스트에 담긴 데이터프레임들을 하나로 합치기
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    # 4. 합쳐진 결과를 새로운 CSV 파일로 저장
    combined_df.to_csv(
        output_path() / 'slope_lables.csv', index=False, encoding='utf-8-sig'
    )
