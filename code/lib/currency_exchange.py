# -*- coding: utf-8 -*-


def currency_exchange(df, money_column, currency_column):
    # 汇率转换
    for i, row in df.iterrows():
        if row[currency_column] == '人民币元':
            exchanged = row[money_column]
        elif row[currency_column] == '美元':
            exchanged = row[money_column] * 6.85
        elif row[currency_column] == '欧元':
            exchanged = row[money_column] * 8.06
        elif row[currency_column] == '香港元':
            exchanged = row[money_column] * 0.87
        elif row[currency_column] == '瑞士法郎':
            exchanged = row[money_column] * 7.14
        elif row[currency_column] == '日元':
            exchanged = row[money_column] * 0.06
        elif row[currency_column] == '英镑':
            exchanged = row[money_column] * 9.08
        elif row[currency_column] == '新加坡元':
            exchanged = row[money_column] * 5.01
        else:
            exchanged = row[money_column]
        df.at[i, money_column] = exchanged


if __name__ == '__main__':
    pass
