import pandas as pd

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})

print(df)


print(pd.get_dummies(df, prefix=['A'], columns=['A']))
