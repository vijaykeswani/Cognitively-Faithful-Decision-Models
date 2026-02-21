import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import statsmodels.api as sm
import scipy.stats as sci
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
# from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
# from tensorflow.keras.models import Sequential
# import tensorflow
# from tensorflow.keras import backend as K



def get_data(study="one"):
    if study == "one":
        repids = ['easy1', 'easy2', 'easy3', 'hard1', 'hard2', 'hard3']
    else:
        repids = ['easy_1', 'easy_2', 'easy_3', 'hard_1', 'hard_2', 'hard_3']

    # df[idvar] = df[[idvar, 'sessionid']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    if study == "one":
        data = pd.read_csv('data/ipmdata0316722.csv').drop('Unnamed: 0',axis=1)
        demos = pd.read_csv('data/ipmdemo.csv').drop('Unnamed: 0', axis=1)
        demos = demos.drop(demos.loc[(demos['uid']==13) & (demos['econStatus'] == 26)].index)    

        outlier_threshold = np.std(data['secElasped'])*3
        mean = np.mean(data['secElasped'])
        clean_data = data.loc[data['secElasped'] <= outlier_threshold].copy()
        
        def f(x):
            return x[0] - x[1]
        data['alcodiff'] = data[['lalco','ralco']].apply(f, axis=1)
        data['depdiff'] = data[['ldep','rdep']].apply(f,axis=1)
        data['lifediff'] = data[['llife','rlife']].apply(f,axis=1)
        data['crimdiff'] = data[['lcrim','rcrim']].apply(f,axis=1)
        clean_data['alcodiff'] = clean_data[['lalco','ralco']].apply(f, axis=1)
        clean_data['depdiff'] = clean_data[['ldep','rdep']].apply(f,axis=1)
        clean_data['lifediff'] = clean_data[['llife','rlife']].apply(f,axis=1)
        clean_data['crimdiff'] = clean_data[['lcrim','rcrim']].apply(f,axis=1)
        
        full = data.merge(demos, on='uid', how='left')
        clean_full = clean_data.merge(demos, on='uid', how='left')
        
        for user in full['uid'].unique():
            for sesh in full.loc[full['uid'] == user,'sessionid'].unique():
                if len(full.loc[full['sessionid'] == sesh]) < 60:
                    full = full.loc[full['sessionid'] != sesh]
        for user in clean_full['uid'].unique():
            for sesh in clean_full.loc[clean_full['uid'] == user,'sessionid'].unique():
                if len(clean_full.loc[clean_full['sessionid'] == sesh]) < 60:
                    clean_full = clean_full.loc[clean_full['sessionid'] != sesh]
        
        
        session_list = full.loc[full['attfailed'] == 1,'sessionid']
        for session in session_list:
            full = full.loc[full['sessionid'] != session]
            
        session_list = clean_full.loc[clean_full['attfailed'] == 1,'sessionid']
        for session in session_list:
            clean_full = clean_full.loc[clean_full['sessionid'] != session]
        
        complete_data = full.copy()
        clean_complete_data = clean_full.copy()
        for user in set(full['uid']):
            if len(full.loc[full['uid'] == user]) < 300:
                complete_data = complete_data.loc[full['uid'] != user]
            if len(clean_full.loc[clean_full['uid'] == user]) < 300:
                clean_complete_data = clean_complete_data.loc[clean_full['uid'] != user]
        complete_data = complete_data.drop('attfailed',axis=1).reset_index()
        clean_complete_data = clean_complete_data.drop('attfailed',axis=1).reset_index()
        
        full = full.sort_values('pairid')
        clean_full = clean_full.sort_values('pairid')
        users = set(full['uid'])
        clean_users = set(clean_full['uid'])
        complete_users = set(complete_data['uid'])
        clean_complete_users = set(clean_complete_data['uid'])
        questions = ['easy1','easy2','easy3','hard1','hard2','hard3']
        
        # columns = ['lalco', 'ldep', 'llife', 'lcrim', 'ralco', 'rdep', 'rlife', 'rcrim', 'chosen', 'alcodiff', 'depdiff', 'lifediff', 'crimdiff']
        columns = ['uid', 'pairid', 'lalco', 'ldep', 'llife', 'lcrim', 'ralco', 'rdep', 'rlife', 'rcrim', 'secElasped',
           'chosen', 'order', 'alcodiff', 'depdiff', 'lifediff', 'crimdiff', 'gender', 'age',
           'numberOfChildren', 'employment', 'maritalStatus', 'ethnicity',
           'political', 'religion', 'religiousness', 'econStatus']
        
        final_df = full[columns].copy()
        clean_final_df = clean_full[columns].copy()
        comp_final_df = complete_data[columns].copy()
        comp_clean_final_df = clean_complete_data[columns].copy()        

        # final_df = full.copy()
        # clean_final_df = clean_full.copy()
        # comp_final_df = complete_data.copy()
        # comp_clean_final_df = clean_complete_data.copy()        
        
        final_df['chosen'] = final_df['chosen'].map({1:0, 0:1})
        clean_final_df['chosen'] = clean_final_df['chosen'].map({1:0, 0:1})
        comp_final_df['chosen'] = comp_final_df['chosen'].map({1:0, 0:1})
        comp_clean_final_df['chosen'] = comp_clean_final_df['chosen'].map({1:0, 0:1})
        return comp_clean_final_df

    if study == "two":
        data = pd.read_csv('./data/final_exp_2_data.csv')
        demos = pd.read_csv('./data/ipm2-demo-mapped.csv').drop('Unnamed: 0', axis=1)        

        outlier_threshold = np.std(data['secElasped'])*3
        def f(x):
            return x[0] - x[1]

        clean_data = data.loc[data['secElasped'] <= outlier_threshold].copy()
        data['eldepdiff'] = data[['l_elderlyDep','r_elderlyDep']].apply(f, axis=1)
        data['lifediff'] = data[['l_lifeYearsGained','r_lifeYearsGained']].apply(f, axis=1)
        data['obesdiff'] = data[['l_obesity','r_obesity']].apply(f,axis=1)
        data['workdiff'] = data[['l_weeklyWorkhours','r_weeklyWorkhours']].apply(f,axis=1)
        data['waitdiff'] = data[['l_yearsWaiting','r_yearsWaiting']].apply(f,axis=1)
        clean_data['eldepdiff'] = clean_data[['l_elderlyDep','r_elderlyDep']].apply(f, axis=1)
        clean_data['lifediff'] = clean_data[['l_lifeYearsGained','r_lifeYearsGained']].apply(f, axis=1)
        clean_data['obesdiff'] = clean_data[['l_obesity','r_obesity']].apply(f,axis=1)
        clean_data['workdiff'] = clean_data[['l_weeklyWorkhours','r_weeklyWorkhours']].apply(f,axis=1)
        clean_data['waitdiff'] = clean_data[['l_yearsWaiting','r_yearsWaiting']].apply(f,axis=1)
        
        data['attfailed'] = 0
        data.loc[data['pairid']=='att_1','attfailed'] = data.loc[data['pairid']=='att_1',['l_lifeYearsGained','chosen']].apply(lambda x: 1 if (((x[0] == -1) & (x[1] == 0)) | ((x[0] != -1) & (x[1] == 1))) else 0, axis=1)
        data.loc[data['pairid']=='att_2','attfailed'] = data.loc[data['pairid']=='att_2',['l_lifeYearsGained','chosen']].apply(lambda x: 1 if (((x[0] == -1) & (x[1] == 0)) | ((x[0] != -1) & (x[1] == 1))) else 0, axis=1)
        session_list = data.loc[data['attfailed'] == 1,'sessionid']
        
        clean_data['attfailed'] = 0
        clean_data.loc[clean_data['pairid']=='att_1','attfailed'] = clean_data.loc[clean_data['pairid']=='att_1',['l_lifeYearsGained','chosen']].apply(lambda x: 1 if (((x[0] == -1) & (x[1] == 0)) | ((x[0] != -1) & (x[1] == 1))) else 0, axis=1)
        clean_data.loc[clean_data['pairid']=='att_2','attfailed'] = clean_data.loc[clean_data['pairid']=='att_2',['l_lifeYearsGained','chosen']].apply(lambda x: 1 if (((x[0] == -1) & (x[1] == 0)) | ((x[0] != -1) & (x[1] == 1))) else 0, axis=1)
        clean_session_list = clean_data.loc[clean_data['attfailed'] == 1,'sessionid']
        
        full = data.merge(demos, on='id', how='left')
        clean_full = clean_data.merge(demos, on='id', how='left')
        
        for user in full['id'].unique():
            for sesh in full.loc[full['id'] == user,'sessionid'].unique():
                if len(full.loc[full['sessionid'] == sesh]) < 60:
                    full = full.loc[full['sessionid'] != sesh]
        for user in clean_full['id'].unique():
            for sesh in clean_full.loc[clean_full['id'] == user,'sessionid'].unique():
                if len(clean_full.loc[clean_full['sessionid'] == sesh]) < 60:
                    clean_full = clean_full.loc[clean_full['sessionid'] != sesh]
        
        for session in session_list:
            full = full.loc[full['sessionid'] != session]
            
        for session in clean_session_list:
            clean_full = clean_full.loc[clean_full['sessionid'] != session]
        
        complete_data = full.copy()
        clean_complete_data = clean_full.copy()
        for user in set(full['id']):
            if len(full.loc[full['id'] == user]) < 300:
                complete_data = complete_data.loc[full['id'] != user]
            if len(clean_full.loc[clean_full['id'] == user]) < 300:
                clean_complete_data = clean_complete_data.loc[clean_full['id'] != user]
        complete_data = complete_data.drop('attfailed',axis=1).reset_index()
        clean_complete_data = clean_complete_data.drop('attfailed',axis=1).reset_index()
        
        full = full.sort_values('pairid')
        clean_full = clean_full.sort_values('pairid')
        users = set(full['id'])
        clean_users = set(clean_full['id'])
        complete_users = set(complete_data['id'])
        clean_complete_users = set(clean_complete_data['id'])
        questions = ['easy_1','easy_2','easy_3','hard_1','hard_2','hard_3']
        
        columns = ['id', 'secElasped', 'chosen', 'order', 'sessionid', 'l_elderlyDep', 'l_lifeYearsGained', 'l_obesity', 
        'l_weeklyWorkhours', 'l_yearsWaiting', 'r_elderlyDep', 'r_lifeYearsGained', 'r_obesity', 'r_weeklyWorkhours', 
        'r_yearsWaiting', 'eldepdiff', 'lifediff', 'obesdiff', 'workdiff', 'waitdiff', 'pairid']

        final_df = full[columns].copy()
        clean_final_df = clean_full[columns].copy()
        comp_final_df = complete_data[columns].copy()
        comp_clean_final_df = clean_complete_data[columns].copy()

        final_df['chosen'] = final_df['chosen'].map({1:0, 0:1})
        clean_final_df['chosen'] = clean_final_df['chosen'].map({1:0, 0:1})
        comp_final_df['chosen'] = comp_final_df['chosen'].map({1:0, 0:1})
        comp_clean_final_df['chosen'] = comp_clean_final_df['chosen'].map({1:0, 0:1})
        
        return comp_clean_final_df


    if study=="simulated":
        N = 1000
        lprefix = "l_"
        rprefix = "r_"

        feats = ['dependents', 'life_gained', 'years_waiting', 'crimes']
        feat_values = {'dependents': list(range(6)),  
                    # 'life_gained': [1, 5, 10, 15, 20, 25, 30], 
                    'life_gained': list(range(1,30, 5)), 
                    'years_waiting': list(range(1,11)),
                    'crimes': list(range(4))}

        df_all = pd.DataFrame()

        m=10
        for r in range(m):
            df = dgp1(N, lprefix, rprefix, feats, feat_values, r*m+1)
            df_all = pd.concat([df_all, df], ignore_index=True, sort=False)

            df = dgp2(N, lprefix, rprefix, feats, feat_values, r*m+2)
            df_all = pd.concat([df_all, df], ignore_index=True, sort=False)

            df = dgp3(N, lprefix, rprefix, feats, feat_values, r*m+3)
            df_all = pd.concat([df_all, df], ignore_index=True, sort=False)

            df = dgp4(N, lprefix, rprefix, feats, feat_values, r*m+4)
            df_all = pd.concat([df_all, df], ignore_index=True, sort=False)

            df = dgp5(N, lprefix, rprefix, feats, feat_values, r*m+5)
            df_all = pd.concat([df_all, df], ignore_index=True, sort=False)




        return df_all

def dgp1(N, lprefix, rprefix, feats, feat_values, r):
    df = pd.DataFrame()
    for f in feats:
        df[lprefix+f] = np.random.choice(feat_values[f], size=N)
        df[rprefix+f] = np.random.choice(feat_values[f], size=N)
            
    df['id'] = r
    df['chosen'] = 1
    for i, row in df.iterrows():
        points_A, points_B = [], []
        if row[lprefix+'life_gained'] > row[rprefix+'life_gained']:
            pts = 1, 0
        elif row[lprefix+'life_gained'] < row[rprefix+'life_gained']:
            pts = 0, 1
        else:
            pts = 0, 0 
        points_A.append(pts[0])
        points_B.append(pts[1])

        if row[lprefix+'dependents']>0:
            points_A.append(1)
        if row[rprefix+'dependents']>0:
            points_B.append(1)
                
        if row[lprefix+'years_waiting']>6:
            points_A.append(1)
        if row[rprefix+'years_waiting']>6:
            points_B.append(1)
                
        total = sum(points_A) - sum(points_B) + np.random.normal(0, 1)
        chosen = int(total > 0)


        df.at[i, 'chosen'] = chosen
    return df

def dgp2(N, lprefix, rprefix, feats, feat_values, r):
    df = pd.DataFrame()
    for f in feats:
        df[lprefix+f] = np.random.choice(feat_values[f], size=N)
        df[rprefix+f] = np.random.choice(feat_values[f], size=N)
            
    df['id'] = r
    df['chosen'] = 1
    for i, row in df.iterrows():
        points_A, points_B = [], []

        total = 0 
        if row[lprefix+'dependents']>0:
            total += 1
        if row[rprefix+'dependents']>0:
            total -= 1

        if total == 0:
            total = np.sign(row[lprefix+'life_gained'] - row[rprefix+'life_gained'])
                                            
        if total == 0:
            total = np.sign(row[lprefix+'years_waiting'] - row[rprefix+'years_waiting'])

        if total == 0:
            total = np.random.normal(0,1)

        chosen = int(total > 0)


        df.at[i, 'chosen'] = chosen
    return df

def dgp3(N, lprefix, rprefix, feats, feat_values, r):
    df = pd.DataFrame()
    for f in feats:
        df[lprefix+f] = np.random.choice(feat_values[f], size=N)
        df[rprefix+f] = np.random.choice(feat_values[f], size=N)
            
    df['id'] = r
    df['chosen'] = 1
    for i, row in df.iterrows():
        points_A, points_B = [], []
        llife, rlife = int(np.log(row[lprefix+'life_gained'])), int(np.log(row[rprefix+'life_gained']))
        pts = [llife, rlife]
        # if llife > rlife:
        #     pts = 1, 0
        # elif llife < rlife:
        #     pts = 0, 1
        # else:
        #     pts = 0, 0 
        points_A.append(pts[0])
        points_B.append(pts[1])

        points_A.append(row[lprefix+'dependents'])
        points_B.append(row[rprefix+'dependents'])


        # if row[lprefix+'dependents']>0:
        #     points_A.append(1)
        # if row[rprefix+'dependents']>0:
        #     points_B.append(1)
                
        if row[lprefix+'years_waiting']>5:
            points_A.append(1)
        if row[rprefix+'years_waiting']>5:
            points_B.append(1)
                
        total = sum(points_A) - sum(points_B) 
        # if total == 0:
            # total = -1 if row[lprefix+'crimes'] > row[rprefix+'crimes'] else 1            
        
        if total == 0:
            total = np.random.normal(0,1)
        chosen = int(total > 0)


        df.at[i, 'chosen'] = chosen
    return df

def dgp4(N, lprefix, rprefix, feats, feat_values, r):
    df = pd.DataFrame()
    for f in feats:
        df[lprefix+f] = np.random.choice(feat_values[f], size=N)
        df[rprefix+f] = np.random.choice(feat_values[f], size=N)
            
    df['id'] = r
    df['chosen'] = 1
    for i, row in df.iterrows():
        points_A, points_B = [], []

        llife, rlife = int(np.log(row[lprefix+'life_gained'])), int(np.log(row[rprefix+'life_gained']))
        total = 0 
        total = np.sign(llife - rlife)

        if total == 0:
            total = np.sign(row[lprefix+'dependents'] - row[rprefix+'dependents'])
                                            
        if total == 0:
            total = np.sign(row[lprefix+'years_waiting'] - row[rprefix+'years_waiting'])

        if total == 0:
            total = np.random.normal(0,1)

        chosen = int(total > 0)


        df.at[i, 'chosen'] = chosen
    return df

def dgp5(N, lprefix, rprefix, feats, feat_values, r):
    df = pd.DataFrame()
    for f in feats:
        df[lprefix+f] = np.random.choice(feat_values[f], size=N)
        df[rprefix+f] = np.random.choice(feat_values[f], size=N)
            
    df['id'] = r
    df['chosen'] = 1
    for i, row in df.iterrows():
        points_A, points_B = [], []

        total = 0
        total += np.sign(row[lprefix+'dependents'] - row[rprefix+'dependents'])
        total += np.sign(row[lprefix+'life_gained'] - row[rprefix+'life_gained'])
        total += np.sign(row[lprefix+'years_waiting'] - row[rprefix+'years_waiting'])
        total += np.sign(row[lprefix+'crimes'] - row[rprefix+'crimes'])

        if total == 0:
            total = np.random.normal(0,1)

        chosen = int(total > 0)

        df.at[i, 'chosen'] = chosen
    return df

