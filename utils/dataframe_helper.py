import pandas as pd
import numpy as np



def split_df(df, n):
    """
    Chia DataFrame thành n phần bằng nhau.
    Nếu số hàng không chia hết cho n, các phần đầu tiên sẽ có nhiều hàng hơn.
    Args:
        df (pd.DataFrame): DataFrame cần chia.
        n (int): Số phần muốn chia.
    Returns:
        list: Danh sách các DataFrame nhỏ.
    """
    if n <= 0: raise ValueError("Số phần phải lớn hơn 0")
    remainder = len(df) % n
    parts = []

    start = 0
    for i in range(n):
        end = start + len(df)//n + (1 if i < remainder else 0)
        parts.append(df.iloc[start:end])
        start = end

    return parts

def merge_dataframes(parts):
    return pd.concat(parts, ignore_index=True).drop_duplicates()
