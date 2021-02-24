import pandas as pd

def calcor(path):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    data = pd.read_csv(path)
    cor = data.corr()

    return cor

if __name__ == '__main__':
    path = './train.csv'
    cor = calcor(path)

    print(cor)
