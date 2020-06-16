import pandas as pd
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report

# todo get amount of unique ids and their respective starting index

""" Is delayed recall a better predictor than immediate recall"""
# full dataset without alterations
df_a = pd.read_csv('ADNIAVLTAmyloidClassification.csv')

# setup to determine gender stats
g_gender = df_a.groupby('sex')
mvf = g_gender.apply(lambda x: x['rid'].unique())
subject_num = len(df_a['rid'].unique())
male_num = mvf.iloc[0].size
female_num = mvf.iloc[1].size

# setup to determine gender amyloid ratio
amg_ratio = df_a.groupby('abeta6mcut')
pamr = amg_ratio.apply(lambda x: x['rid'].unique())
num_pos = pamr.iloc[1].size
test_group = df_a.groupby(['abeta6mcut', 'sex'])
test_ap = test_group.apply(lambda x: x['rid'].unique())
num_pos_m = len(test_ap.reset_index().iloc[2][0])

# setup to determine genotype stats
g_geno = df_a.groupby('genotype')
guniq = g_geno.apply(lambda x: x['rid'].unique())
g_2_num = guniq.iloc[0].size
g_3_num = guniq.iloc[1].size
g_4_num = guniq.iloc[2].size
g_5_num = guniq.iloc[3].size
g_6_num = guniq.iloc[4].size

# setup to determine genotype amyloid ratio
geno_group = df_a.groupby(['abeta6mcut', 'genotype'])
geno_ap = geno_group.apply(lambda x: x['rid'].unique())
geno_df = geno_ap.reset_index()
num_pos_g2 = len(geno_df.iloc[4][0])
num_pos_g3 = len(geno_df.iloc[5][0])
num_pos_g4 = len(geno_df.iloc[6][0])
num_pos_g5 = len(geno_df.iloc[7][0])
num_pos_g6 = len(geno_df.iloc[8][0])

d_recall = ['t6sum', 't7sum']
i_recall = ['t1sum', 't2sum', 't3sum', 't4sum', 't5sum']
samples = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54,
           56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92]
drop_col = ['abeta6mcut']


def p_output(df, expected, trials=100):
    scores = []
    clf = tree.DecisionTreeClassifier()
    test_data = df.loc[samples, :]
    train_data = df.drop(samples, axis=0)
    for i in range(trials):
        clf.fit(train_data, train_positive)
        temp = clf.predict(test_data)
        # report = classification_report(expected, temp)
        # print(report)
        diff = expected - temp
        score = np.count_nonzero(diff == 0) / len(expected)
        scores.append(score)
    return scores

# def p_output_time(expected, trials=100):
#     scores = []
#     clf = tree.DecisionTreeClassifier()
#     df_a_fin = df_a2.loc[df_a2['month'] != 12].drop(['rid'], axis=1)
#     df_fin_test = df_a2.loc[df_a2['month'] == 12].drop(['rid', 'age'], axis=1)
#     for i in range(trials):
#         clf.fit(df_a_fin, df_a.loc[df_a['month'] != 12]['abeta6mcut'])
#         temp = clf.predict(df_fin_test)
#         # report = classification_report(expected, temp)
#         # print(report)
#         diff = expected - temp
#         score = np.count_nonzero(diff == 0) / len(expected)
#         scores.append(score)
#     return scores


# stats we are trying to predict removed from dataframe copy
df_a2 = df_a.drop(drop_col, 1)
# delayed recall specific results dataframe
df_a_dr = df_a2.loc[:, d_recall]
# immediate recall specific results dataframe
df_a_ir = df_a2.loc[:, i_recall]
# mental impairment dataframe
df_a_mi = df_a2.loc[:, ['dx']]
# genotype dataframe
df_a_geno = df_a2.loc[:, ['genotype']]
# status of amyloid positivity
test_positive = df_a.loc[samples, ['abeta6mcut']]
# training dataset for amyloid positivity status
train_positive = df_a['abeta6mcut'].drop(samples, axis=0)
expected_array = np.asarray(test_positive['abeta6mcut'].tolist())
e_arr_12 = np.asarray(df_a.loc[df_a['month'] == 12]['abeta6mcut'].tolist())

# variables for histogram creation
max_age = df_a['age'].max()
min_age = df_a['age'].min()
# mu = df_a['age'].mean()
# sigma = df_a['age'].std()
# x = df_a['age']
# num_bins = 5
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# plt.show()

print(f'Number of Subjects: {subject_num}', end="\n\n")

print(f'Max subject age: {max_age}')
print(f'Min subject age: {min_age}', end="\n\n")

print(f'Number of Male Subjects: {male_num}')
print(f'Number of Female Subjects: {female_num}')
print(f'Number of Positive Subjects: {num_pos}')
print(f'Positive Subject Ratio: {num_pos / subject_num}')
print(f'Male Positive to Total Male Ratio: {num_pos_m / male_num}')
print(f'Female Positive to Total Female Ratio: {(num_pos - num_pos_m) / female_num}')
print(f'Positive Males to Total Positive Ratio: {num_pos_m / num_pos}')
print(f'Positive Females to Total Positive Ratio: {(num_pos - num_pos_m) / num_pos}')
print(f'Positive Males to Total Ratio: {num_pos_m / subject_num}')
print(f'Positive Females to Total Ratio: {(num_pos - num_pos_m) / subject_num}', end='\n\n')

print(f'Participant ratio g2: {g_2_num/subject_num}')
print(f'Participant ratio g3: {g_3_num/subject_num}')
print(f'Participant ratio g4: {g_4_num/subject_num}')
print(f'Participant ratio g5: {g_5_num/subject_num}')
print(f'Participant ratio g6: {g_6_num/subject_num}')
print(f'Percentage of g2 positive: {num_pos_g2/g_2_num}')
print(f'Percentage of g3 positive: {num_pos_g3/g_3_num}')
print(f'Percentage of g4 positive: {num_pos_g4/g_4_num}')
print(f'Percentage of g5 positive: {num_pos_g5/g_5_num}')
print(f'Percentage of g6 positive: {num_pos_g6/g_6_num}', end='\n\n')

print('---------- Expected Results --------------')
print(expected_array)

print('---------- Test 1: Delayed Recall --------------')
dr_results = p_output(df_a_dr, expected_array)
print(f'Test Percentage Correct: {dr_results}')
print('Average: ' + str(sum(dr_results) / len(dr_results)))

print('---------- Test 2: Immediate Recall --------------')
ir_results = p_output(df_a_ir, expected_array)
print(f'Test Percentage Correct: {ir_results}')
print('Average: ' + str(sum(ir_results) / len(ir_results)))

print('---------- Test 3: Mental Impairment --------------')
mi_results = p_output(df_a_mi, expected_array)
print(f'Test Percentage Correct: {mi_results}')
print('Average: ' + str(sum(mi_results) / len(mi_results)))

print('---------- Test 4: Genotype --------------')
ge_results = p_output(df_a_geno, expected_array)
print(f'Test Percentage Correct: {ge_results}')
print('Average: ' + str(sum(ge_results) / len(ge_results)))


# print('---------- Test 5: Time Predictions --------------')
# # print(e_arr_12)
# pt_results = p_output_time(e_arr_12)
# print(f'Test Percentage Correct: {pt_results}')
# print('Average: ' + str(sum(pt_results) / len(pt_results)))
