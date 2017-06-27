import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

files = [
    # 'answer_0_5964.csv',
    # 'answer_0_6254.csv',
    # 'answer_0_6296.csv',
    # 'answer_0_6395.csv',
    # 'answer_0_6424.csv',
    'answer_0_6614.csv',
    # 'answer_0_6628.csv',
    # 'answer_0_6636.csv',
    # 'answer_0_6657.csv',
    # 'answer_0_6664.csv',
    'answer_0_6685.csv',
    # 'answer_0_6699.csv',
    'answer_0_6706.csv',
    # 'answer_0_6720.csv',
    'answer_0_6734.csv',
    'answer_0_6742.csv',
    'answer_0_6749.csv',
    'answer_0_6756.csv',
    'answer_0_6763.csv',
]

df = pd.DataFrame()
for file_ in files:
    answer = pd.DataFrame(pd.read_csv('answers/' + file_, names=['answers']))
    df[file_[7:13]] = answer['answers'].astype(np.int64)

vote_answer = []
for pos in range(len(df['0_6756'])):
    row_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    for column in files:
        row_dict[str(df[column[7:13]].iloc[pos])] += 1
    best_answer_count = 0
    for key in row_dict.keys():
        if row_dict[key] > best_answer_count:
            best_answer_count = row_dict[key]
            best_answer = int(key)
    vote_answer.append(best_answer)

df['votes'] = vote_answer

df['diff'] = df['votes'] - df['0_6763']
print(df[df['diff'] != 0])
print(df[df['diff'] != 0].shape)

df['votes'].to_csv('data/answer.csv', index=False)





