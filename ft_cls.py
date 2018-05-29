import pandas as pd
import fasttext as ft
import os
import re
from scipy.stats import bernoulli
import seaborn as sns
import matplotlib.pyplot as plt

# converts list of lists of tuple to list of tuples 
def flat(array):
    result = []
    for i in range(len(array)):
        if type(array[i]) == list:
            for j in flat(array[i]):
                result.append(j)
        else:
            result.append(array[i])
    return result

# helper for collecting a sample of comments for a given ns and year from
def load_no_bot_no_admin(ns, year,prob = 0.1):
    dfs = []

    data_dir = "/corpus_file_path/comments_%s_%d" % (ns, year)
    for _, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if re.match("chunk_\d*.tsv", filename):
                df = pd.read_csv(os.path.join(data_dir, filename), sep="\t")
                df['include'] = bernoulli.rvs(prob, size=df.shape[0])
                df = df.query("bot == 0 and admin == 0 and include == 1")
                dfs.append(df)

    sample = pd.concat(dfs)
    sample['ns'] = ns
    sample['year'] = year

    return sample
# MAIN PROGRAMM
pd.set_option('max_colwidth', 5000)

comments = pd.read_csv('/personal_attack_file_path/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('/personal_attack_file_path/attack_annotations.tsv', sep = '\t')

print(len(annotations['rev_id'].unique()))

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments
comments['attack'] = labels

# remove newline and tab tokens from comments dataframe
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

# create labels from attack column by concatanating values with '__label__'
comments.loc[comments['attack'] == True, 'attack'] = '__label__Attack'
comments.loc[comments['attack'] == False, 'attack'] = '__label__Neutral'

# create train, validation and test sets of personal attacks
train_comments = comments.query("split=='train'")
valid_comments = comments.query("split=='dev'")
test_comments = comments.query("split=='test'")

# training set
with open('/train_file_path/train.txt', 'w') as f:
    f.write(train_comments[['attack', 'comment']].to_csv(sep=' ', chunksize=1000, header=False, index=False))

# validation set
with open('/valid_file_path/valid.txt', 'w') as f:
    f.write(valid_comments[['attack', 'comment']].to_csv(sep=' ', chunksize=1000, header=False, index=False))

# test set
with open('/test_file_path/test.txt', 'w') as f:
    f.write(test_comments[['attack', 'comment']].to_csv(sep=' ', chunksize=1000, header=False, index=False))

# best arguments are lr = 0.5, minn = 2, maxn = 2
classifier = ft.supervised('/train_file_path/train.txt', '/model_path/model', minn = 2, maxn = 4, lr = 0.5)

result_v = classifier.test('/valid_file_path/valid.txt')
result_t = classifier.test('/test_file_path/test.txt')

# print appropriate scores and number of examples
print('P@1:\t{0:.3f}\t{1:.3f}'.format(result_v.precision, result_t.precision))
print('R@1:\t{0:.3f}\t{1:.3f}'.format(result_v.recall, result_t.recall))
print('Number of examples:\t{0}\t{1}'.format(result_v.nexamples, result_t.nexamples))

# collect a random sample of comments from 2004 for each namespace
corpus_user = load_no_bot_no_admin('user', 2015)
corpus_article = load_no_bot_no_admin('article', 2015)
corpus = pd.concat([corpus_user, corpus_article])

# Apply model
corpus['comment'] = corpus['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
corpus['comment'] = corpus['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

flat_labels = flat(classifier.predict_proba(corpus['comment']))

# assign personal attack label probabilities to list
attack = []
for item in flat_labels:   # see paper for threshold 0.425
    if (item[1] > 0.425 and item[0] == 'Attack'):
        attack.append(item[1])
    elif (item[1] < 0.575 and item[0] == 'Neutral'):
        attack.append(item[1])
    else:
        attack.append(0)

# join appropriate propabilities with corpus
corpus['attack'] = attack

# plot prevalence per ns
sns.pointplot(x = 'ns', y = 'attack', data = corpus)
plt.ylabel("Attack fraction")
plt.xlabel("Dicussion namespace")
plt.show()
