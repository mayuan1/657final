import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from datetime import datetime
from pandas import Series
from math import ceil
import re

df = pd.read_csv("twitch_data_set.csv", encoding = "ISO-8859-1")

df.drop(columns=['stream created time'], inplace = True)

df.columns = ['stream_ID','current_views','game_name','broadcaster_ID','broadcaster_name','delay_setting','follower_number','partner_status','broadcaster_language','total_views_of_this_broadcaster','language','broadcasters_created_time','playback_bitrate','source_resolution']

cols = ['game_name', 'language', 'delay_setting', 'playback_bitrate', 'source_resolution']

# for col in cols:
#     df['Name'] = df[col]
#     chart = df[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
#     sns.set_style("white")
#     plt.figure(figsize=(12.4, 5))
#     plt.xticks(rotation=90)
#     sns.barplot(x=col, y='Name', data=chart[:30], palette=sns.cubehelix_palette((12 if col == 'Genre' else 30), dark=0.3, light=.85, reverse=True)).set_title(('Game count by '+col), fontsize=16)
#     plt.ylabel('Count', fontsize=14)
#     plt.xlabel('')
#     plt.show()

df = df[df.game_name != '-1']
df = df[df.language != '-1']
df = df[df.playback_bitrate != -1]
# df = df[df.current_views > 100]




# for col in cols:
#     df['Name'] = df[col]
#     chart = df[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
#     sns.set_style("white")
#     plt.figure(figsize=(12.4, 5))
#     plt.xticks(rotation=90)
#     sns.barplot(x=col, y='Name', data=chart[:30], palette=sns.cubehelix_palette((12 if col == 'Genre' else 30), dark=0.3, light=.85, reverse=True)).set_title(('Game count by '+col), fontsize=16)
#     plt.ylabel('Count', fontsize=14)
#     plt.xlabel('')
#     #plt.xticks(rotation=45) 
#     plt.tight_layout()
#     plt.show()

df = df[df.current_views > 5]

# for col in cols:
#     df['Name'] = df[col]
#     chart = df[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
#     sns.set_style("white")
#     plt.figure(figsize=(12.4, 5))
#     plt.xticks(rotation=90)
#     sns.barplot(x=col, y='Name', data=chart[:30], palette=sns.cubehelix_palette((12 if col == 'Genre' else 30), dark=0.3, light=.85, reverse=True)).set_title(('Game count by '+col), fontsize=16)
#     plt.ylabel('Count', fontsize=14)
#     plt.xlabel('')
#     plt.show()

df['broadcasters_created_time'] = df['broadcasters_created_time'].map(lambda x: re.sub(r'\W+', '', x))
df['broadcasters_created_time'] = df['broadcasters_created_time'].astype(str).str[0:6]
df[['broadcasters_created_time']] = df[['broadcasters_created_time']].apply(pd.to_numeric)
df['source_resolution'] = df['source_resolution'].map(lambda x: re.sub(r'\[a-z]+', '', x))

# print(df[:5])

def game_rename(name):
    if name == "League of Legends":
        return 'LOL'
    elif name == "Grand Theft Auto V":
        return 'GTA V'
    elif name ==  "StarCraft II: Heart of the Swarm":
        return 'StarCraft II'
    elif name ==  "World of Warcraft: Warlords of Draenor":
        return 'WOW'
    elif name ==  "Call of Duty: Advanced Warfare":
        return 'COD:AW'
    elif name ==  "Counter-Strike: Global Offensive":
        return 'CS: GO'
    else:
        return name

df['game_name'] = df['game_name'].apply(lambda x: game_rename(x))

# cols = ['game_name', 'language']
# for col in cols:
#     uniques = df[col].value_counts().keys()
#     uniques_dict = {}
#     ct = 0
#     for i in uniques:
#         uniques_dict[i] = ct
#         ct += 1

#     for k, v in uniques_dict.items():
#         df.loc[df[col] == k, col] = v

# df1 = df[['game_name', 'language', 'broadcasters_created_time','current_views','follower_number']]
# df1 = df1.dropna().reset_index(drop=True)
# df1 = df1.astype('float64')

# mask = np.zeros_like(df1.corr())
# mask[np.triu_indices_from(mask)] = True

# cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470
# with sns.axes_style("white"):
#     fig, ax = plt.subplots(1,1, figsize=(10,10))
#     ax = sns.heatmap(df1.corr(), mask=mask, vmax=0.2, square=True, annot=True, fmt=".3f", cmap=cmap)

# plt.yticks(rotation=0)
# plt.xticks(rotation=10) 
# plt.tight_layout()
# plt.show()

# # fig, ax = plt.subplots(1,1, figsize=(12,5))
# # sns.regplot(x="game_name", y="current_views", data=df1.loc[df1.follower_number <= 10000])
# # plt.show()

# follower = df['current_views'].values
# print(max(follower))
# print(min(follower))

def follower_group(follower):
    if follower >= 500000:
        return '> 500'
    elif follower >= 100000:
        return '100 - 500'
    elif follower >= 50000:
        return '50 - 100'
    elif follower >= 10000:
        return '10 - 50'
    elif follower >= 5000:
        return '5 - 10'
    elif follower >= 1000:
        return '1 - 5'
    else:
        return '0 - 1'

dfh = df.dropna(subset=['follower_number']).reset_index(drop=True)
dfh['follower_group'] = dfh['follower_number'].apply(lambda x: follower_group(x))

def create_group(time):
    if time <= 201106:
        return '> 4'
    elif time <= 201206:
        return '3 - 4'
    elif time <= 201306:
        return '2 - 3'
    elif time <= 201406:
        return '1 - 2'
    else:
        return '0 - 1'

dfh = dfh.dropna(subset=['broadcasters_created_time']).reset_index(drop=True)
dfh['create_group'] = dfh['broadcasters_created_time'].apply(lambda x: create_group(x))

# print(dfh[:5])

# dfh = dfh.loc[dfh['game_name'].isin(["Dying Light","League of Legends","Grand Theft Auto V","StarCraft II: Heart of the Swarm","World of Warcraft: Warlords of Draenor","Dota 2","Counter-Strike: Global Offensive","Hearthstone"])]
dfh = dfh.loc[dfh['game_name'].isin(["Dying Light","LOL","GTA V","Destiny","WOW","COD:AW","CS: GO","Minecraft"])]

def in_top(x):
    if x in pack:
        return x
    else:
        pass

cols = ['follower_group', 'create_group']
for col in cols:
    pack = []
    dfh['Name'] = dfh[col]
    top = dfh[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
    for x in top[col]:
        pack.append(x)
    dfh[col] = dfh[col].apply(lambda x: in_top(x))
    dfh_platform = dfh[[col, 'game_name', 'current_views']].groupby([col, 'game_name']).mean().reset_index().pivot(col, 'game_name', 'current_views')
    plt.figure(figsize=(8, 5))
    sns.heatmap(dfh_platform, annot=True, fmt=".2g", linewidths=.5).set_title((' \n'+col+' vs. game name (by mean views) \n'))
    plt.ylabel('')
    plt.xlabel('Game Name \n')
    pack = []
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# dfb = df[['game_name', 'language', 'broadcasters_created_time','current_views','follower_number']]
# dfb = dfb.dropna().reset_index(drop=True)
# df2 = dfb[['game_name', 'language', 'broadcasters_created_time','current_views','follower_number']]
# df2['Hit'] = df2['current_views']
# df2.drop('current_views', axis=1, inplace=True)

# def hit(view):
#     if view >= 1000:
#         return 1
#     else:
#         return 0

# df2['Hit'] = df2['Hit'].apply(lambda x: hit(x))

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
# from sklearn import svm

# from pandas import get_dummies
# df_copy = pd.get_dummies(df2)

# df3 = df_copy
# y = df3['Hit'].values
# df3 = df3.drop(['Hit'],axis=1)
# X = df3.values

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50, random_state=2)


# #####################
# radm = RandomForestClassifier()
# parameter_space = { 
#     'n_estimators': [200, 500, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# clf = GridSearchCV(radm, parameter_space, n_jobs=-1, cv=3)
# clf.fit(Xtrain, ytrain)

# # Best paramete set
# print('Best parameters found:\n', clf.best_params_)

# # All results
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# y_true, y_pred = ytest , clf.predict(Xtest)

# print('Results on the test set:')
# print(classification_report(y_true, y_pred))


# ######################

# # radm = RandomForestClassifier(random_state=2).fit(Xtrain, ytrain)
# # y_val_1 = radm.predict_proba(Xtest)
# # print("Validation accuracy: ", sum(pd.DataFrame(y_val_1).idxmax(axis=1).values
# #                                    == ytest)/len(ytest))

# # all_predictions = radm.predict(Xtest)
# # print(classification_report(ytest, all_predictions))

# # indices = np.argsort(radm.feature_importances_)[::-1]

# # # Print the feature ranking
# # print('Feature ranking (top 10):')

# # for f in range(10):
# #     print('%d. feature %d %s (%f)' % (f+1 , indices[f], df3.columns[indices[f]],
# #                                       radm.feature_importances_[indices[f]]))

# # svm_reg = svm.SVC(C=1.0,probability=True).fit(Xtrain, ytrain)
# # y_val_2 = svm_reg.predict_proba(Xtest)
# # print("Validation accuracy: ", sum(pd.DataFrame(y_val_2).idxmax(axis=1).values
# #                                    == ytest)/len(ytest))

# # all_predictions2 = svm_reg.predict(Xtest)
# # print(classification_report(ytest, all_predictions2))

# # mlp = MLPClassifier(max_iter=100)
# # parameter_space = {
# #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
# #     'activation': ['tanh', 'relu'],
# #     'solver': ['sgd', 'adam'],
# #     'alpha': [0.0001, 0.05],
# #     'learning_rate': ['constant','adaptive'],
# # }

# # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
# # clf.fit(Xtrain, ytrain)

# # # Best paramete set
# # print('Best parameters found:\n', clf.best_params_)

# # # All results
# # means = clf.cv_results_['mean_test_score']
# # stds = clf.cv_results_['std_test_score']
# # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
# #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# # y_true, y_pred = ytest , clf.predict(Xtest)

# # print('Results on the test set:')
# # print(classification_report(y_true, y_pred))

# # svm_reg = MLPClassifier(C=1.0,probability=True).fit(Xtrain, ytrain)
# # y_val_2 = svm_reg.predict_proba(Xtest)
# # print("Validation accuracy: ", sum(pd.DataFrame(y_val_2).idxmax(axis=1).values
# #                                    == ytest)/len(ytest))

# # all_predictions2 = svm_reg.predict(Xtest)
# # print(classification_report(ytest, all_predictions2))