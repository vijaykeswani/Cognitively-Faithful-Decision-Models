import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import sklearn
import sklearn.metrics
import statsmodels.api as sm
import scipy.stats as sci
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import copy
import random

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tqdm.notebook import tqdm
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from functools import *
import choix
import matplotlib.pyplot as plt
import itertools
import math
from scipy.special import expit as sigmoid
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin



### Classifier based on logistic regression - this implementation ensure it is symmetric for pairwise comparisons
class LGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cols, prefixes=["l", "r"], verbose=True):
        self.cols = cols
        self.prefixes = prefixes
        self.verbose = verbose


    def fit(self, X):
        self.X_ = X

        pairs, true_labels, diff_feats = [], [], []
        for row in self.X_.iterrows():
            row = row[1]
            cs1 = [self.prefixes[0] + f for f in self.cols]   
            p1 = dict(zip(self.cols, row[cs1]))
            cs2 = [self.prefixes[1] + f for f in self.cols]
            p2 = dict(zip(self.cols, row[cs2]))
            pairs.append((p1, p2))
            diff_feats.append([p1[f] - p2[f] for f in self.cols])
            true_labels.append(row["chosen"])

        self.pairs_ = pairs
        self.true_labels_ = true_labels

        scalar = StandardScaler()
        diff_feats_transform = scalar.fit_transform(diff_feats)
        self.scalar = scalar

        model = LogisticRegression(penalty=None).fit(diff_feats_transform, true_labels)
        self.logit_model = model
        self.coefs = dict(zip(self.cols, model.coef_[0]))

        if self.verbose:
            print (self.coefs)

        return self
    
    def predict(self, X):
        diff_feats = []
        for row in X.iterrows():
            row = row[1]
            cs1 = [self.prefixes[0] + f for f in self.cols]   
            p1 = dict(zip(self.cols, row[cs1]))
            cs2 = [self.prefixes[1] + f for f in self.cols]
            p2 = dict(zip(self.cols, row[cs2]))
            diff_feats.append([p1[f] - p2[f] for f in self.cols])

        diff_feats_transform = self.scalar.transform(diff_feats)        
        predictions = list(self.logit_model.predict(diff_feats_transform))
        return predictions

    def score(self, X, y=[]):
        preds, true_labels = [], []
        true_labels = [row[1]["chosen"] for row in X.iterrows()]
        preds = self.predict(X)
        
        preds = np.array(preds)
        true_labels = np.array(true_labels)
        acc = np.mean(preds == true_labels)
        return acc       





### Class of decision rules that operates over a single feature
class HeuristicRule:
    def __init__(self, fname, fvalues, rule_vars=[], fsign=1):
        self.fname = fname
        self.fvalues = fvalues
        self.fsign = fsign
        self.rule = {value: rule_vars[i] for i, value in enumerate(self.fvalues)}

    def apply(self, x):
        if x not in self.rule.keys():
            return 0
        # if x == min(self.fvalues):
            # return 0
        return self.fsign*self.rule[x]

    def apply_bin(self, x1, x2, discrete=False):
        if x1 == x2:
            res = 0

        res = self.fsign*int(x1-x2>0)
        return res

    def apply_pair(self, x1, x2, discrete=False):
        if x1 == x2:
            res = 0

        res = self.apply(x1) - self.apply(x2)
        if discrete:
            return res
        else:
            # return sigmoid(res)
            return (res)

    def get_vars(self):
        return list(self.rule.values())

    def __str__(self):
        vals = np.abs(list(self.rule.values()))
        if sum(vals) != 0:
            return self.fname + " : " + str(self.rule)
        return ""
    
    def plot(self, ax):
        vars = self.fsign * np.array(self.get_vars())
        ax.plot(self.fvalues, vars, label=self.fname, linewidth=2.5)
        ax.legend()


### Classifier implementing our two-stage model - learns the decision rules used for generating any given response data
class IsotonicFactorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cols, feature_signs, feature_values, interactions=False, feat_to_diff_feat={},
                 lambda1=0., lambda2=0, prefixes=["l", "r"], cond_feature=None, loss="hinge", verbose=True, label_feat='chosen'):
        self.cols = cols
        self.feature_signs = feature_signs
        self.interactions = interactions
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.feature_values = feature_values
        self.prefixes = prefixes
        self.cond_feature = cond_feature
        self.feat_to_diff_feat = feat_to_diff_feat
        self.ir = None
        self.int_cols = None
        self.verbose = verbose
        self.label_feat = label_feat
        self.loss = loss

    def get_pair(self, row):
        cs1 = [self.prefixes[0] + f for f in self.cols]   
        p1 = dict(zip(self.cols, row[cs1]))
        cs2 = [self.prefixes[1] + f for f in self.cols]
        p2 = dict(zip(self.cols, row[cs2]))

        if self.interactions:

            cond_feature_values = self.feature_values[self.cond_feature]
            new_cols = []
            other_cols = [f for f in self.cols if f != self.cond_feature]
            p1_int, p2_int = {}, {}
            l_val = row[self.prefixes[0] + self.cond_feature]
            r_val = row[self.prefixes[0] + self.cond_feature]

            for f in other_cols:
                for val in cond_feature_values:
                    new_cols.append(f + " | " + self.cond_feature+"="+str(val))
                    if val == l_val:
                        p1_int[f + " | " + self.cond_feature+"="+str(val)] = p1[f]
                    else:
                        p1_int[f + " | " + self.cond_feature+"="+str(val)] = 0

                    if val == r_val:
                        p2_int[f + " | " + self.cond_feature+"="+str(val)] = p2[f]
                    else:
                        p2_int[f + " | " + self.cond_feature+"="+str(val)] = 0
                

            new_cols.append(self.cond_feature)
            p1_int[self.cond_feature] = p1[self.cond_feature]
            p2_int[self.cond_feature] = p2[self.cond_feature]

            self.int_cols = list(new_cols)
            p1, p2 = p1_int, p2_int

        return p1, p2

    def fit(self, X):
        self.X_ = X
        _, _ = self.get_pair(self.X_.iloc[0]) 

        fvals, feat_to_indices, feat_to_fvals = [], {}, {}
        if self.interactions:
            cols_to_use = self.int_cols
        else:
            cols_to_use = self.cols

        for f in cols_to_use:
            start = len(fvals)
            vals = self.feature_values[f.split(' | ')[0]]
            # val_pairs = itertools.combinations(vals, 2)

            # fvals += [f+"_"+str(v1)+","+str(v2) for (v1, v2) in val_pairs]
            fvals += [f+"_"+str(v) for v in vals]
            feat_to_fvals[f] = list(vals)
            feat_to_indices[f] = list(range(start, len(fvals)))


        pairs, true_labels = [], []
        for row in self.X_.iterrows():
            row = row[1]
            p1, p2 = self.get_pair(row)
            pairs.append((p1, p2))
            true_labels.append(row[self.label_feat])

        self.pairs_ = pairs
        self.true_labels_ = true_labels

        def error(rule_vars):
            rules = {f: np.array(rule_vars[feat_to_indices[f]]) for f in cols_to_use}
            rules = [HeuristicRule(feat, self.feature_values[feat.split(' | ')[0]], rules[feat], self.feature_signs[feat.split(' | ')[0]]) for feat in cols_to_use]

            preds, labels = [], []
            indices_train = list(range(len(self.pairs_)))

            for idx in indices_train:
                p1, p2 = self.pairs_[idx]
                pred = self._predict_single((p1, p2), rules)
                pred = sigmoid(pred) if self.loss == "logistic" else pred
                preds.append(pred)

                # preds.append(int(pred > 0.5))
                if self.loss == "hinge":
                    label = 1 if self.true_labels_[idx] == 1 else -1
                else:
                    label = self.true_labels_[idx]
                labels.append(label)

            preds = np.array(preds)
            labels = np.array(labels)

            # print (self.loss, preds, labels)
            if self.loss == "logistic":
                err = sklearn.metrics.log_loss(labels, preds)
            elif self.loss == "hinge":
                err = sklearn.metrics.hinge_loss(labels, preds)
            else:
                raise ValueError("Unknown loss function: {}".format(self.loss))

            reg = 0
            if self.interactions:                
                for f in self.feature_values.keys():
                    rules_f = []
                    for f2 in cols_to_use:
                        if f == f2.split(' | ')[0]:
                            rules_f.append(np.array(rule_vars[feat_to_indices[f2]]))

                    reg += sum([np.linalg.norm(rules_f[i] - rules_f[j], ord=2) for i, j in itertools.combinations(range(len(rules_f)), 2)])

                rules_f = rule_vars[feat_to_indices[cols_to_use[-1]]]
            return err + 0.0001*reg + self.lambda1*np.linalg.norm(rule_vars, ord=1)/len(rule_vars) + self.lambda2*np.linalg.norm(rule_vars, ord=2)/len(rule_vars)

        rule_vars_init = [0]*len(fvals)
        cons_inds = []
        for f in cols_to_use:
            indices = feat_to_indices[f]
            cons_inds += [(i, i-1) for i in indices[1:]]
        
        ineq_cons = {'type': 'ineq', 'fun': lambda u: np.array([u[i] - u[j] for i, j in cons_inds] + [u[i] for i in range(len(rule_vars_init))])}
        eq_cons = {'type': 'eq', 'fun': lambda u: np.sum([u[i] for i in range(len(rule_vars_init))]) - 100}
        bounds = Bounds([0]*len(fvals), [100]*len(fvals))

        # res = minimize(error, rule_vars_init, method='SLSQP', constraints=[ineq_cons, eq_cons], options={'ftol': 1e-7, 'maxiter': 200, 'disp': self.verbose}, )
        res = minimize(error, rule_vars_init, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-7, 'maxiter': 300, 'disp': False}, )

        rule_vars = res.x
        rules = {f: np.array(rule_vars[feat_to_indices[f]]) for f in cols_to_use}
        # rules = {f: [np.round(v) for v in rules[f]] for f in cols_to_use}
        rules = [HeuristicRule(feat, self.feature_values[feat.split(' | ')[0]], rules[feat], self.feature_signs[feat.split(' | ')[0]]) for feat in cols_to_use]
        self.ir = list(rules)

        if self.verbose:
            for r in rules:
                if r.__str__() != "":
                    print (r)
            print ("\n")

        return self
    
    def predict(self, X, rules):
        predictions = [self._predict_single(x, rules) for x in X]
        return predictions

    def _predict_single(self, x, rules, discrete=False):

        p1, p2 = x
        if not self.interactions:
            rule_apps = [rules[i].apply_pair(p1[f], p2[f], discrete) for i, f in enumerate(self.cols)]
        else:        
            rule_apps = [rules[i].apply_pair(p1[f], p2[f], discrete) for i, f in enumerate(self.int_cols)]

        pred = np.mean(rule_apps)

        return pred

    def score(self, X, y=[]):
        preds, true_labels = [], []
        for row in X.iterrows():
            row = row[1]

            p1, p2 = self.get_pair(row)
            pred = self._predict_single((p1, p2), self.ir, discrete=True)
            pred = int(pred > 0)
            preds.append(pred)

            true_labels.append(row[self.label_feat])
        
        preds = np.array(preds)
        true_labels = np.array(true_labels)
        # print (list(zip(preds, true_labels)))
        acc = np.mean(preds == true_labels)
        return acc       




### Class of decision rules operating over pairs of values of any given feature
class PairwiseHeuristicRule:
    def __init__(self, fname, fvalues, rule_vars=[], fsign=1):
        self.fname = fname
        fvalues = sorted(fvalues, reverse=True)
        self.fvalues = fvalues
        self.fsign = fsign

        value_pairs = list(itertools.combinations(fvalues, 2))
        self.rule = {(v1, v2): rule_vars[i] for i, (v1, v2) in enumerate(value_pairs)}

    
    def apply_pair(self, x1, x2, discrete=False):

        if x1 == x2:
            res = 0
        if (x1, x2) in self.rule.keys():
            res = self.fsign*self.rule[(x1, x2)]
        elif (x2, x1) in self.rule.keys():
            res = -self.fsign*self.rule[(x2, x1)]

        if discrete:
            return res
        else:
            return sigmoid(res)
        
                
        
        # return self.fsign*self.rule[(x1, x2)]

    def get_vars(self):
        return list(self.rule.values())

    def __str__(self):
        vals = list(self.rule.values())
        # if sum(vals) != 0:
        return self.fname + " : " + str(self.rule)
        # return ""
    
    def plot(self, ax):
        ax.plot(self.fvalues, self.get_vars(), label=self.fname)
        ax.legend()


class PairwiseIsotonicFactorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cols, feature_signs, feature_values, interactions=False, feat_to_diff_feat={},
                 lambda1=0., lambda2=0, prefixes=["l", "r"], cond_feature=None):
        self.cols = cols
        self.feature_signs = feature_signs
        self.interactions = interactions
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.feature_values = feature_values
        self.prefixes = prefixes
        self.cond_feature = cond_feature
        self.feat_to_diff_feat = feat_to_diff_feat
        self.ir = None
        self.int_cols = None

    def get_pair(self, row):
        cs1 = [self.prefixes[0] + f for f in self.cols]   
        p1 = dict(zip(self.cols, row[cs1]))
        cs2 = [self.prefixes[1] + f for f in self.cols]
        p2 = dict(zip(self.cols, row[cs2]))

        if self.interactions:
            cond_feature_values = self.feature_values[self.cond_feature]
            cond_mult_values = list(itertools.product(cond_feature_values, repeat=2))
            new_cols = []
            other_cols = [f for f in self.cols if f != self.cond_feature]
            p1_int, p2_int = {}, {}
            l_val = row[self.prefixes[0] + self.cond_feature]
            r_val = row[self.prefixes[0] + self.cond_feature]

            for f in other_cols:
                for v1, v2 in cond_mult_values:
                    new_cols.append(f + " | " + self.cond_feature+"="+str(v1)+","+str(v2))

                    if v1 == l_val and v2 == r_val:
                        p1_int[f + " | " + self.cond_feature+"="+str(v1)+","+str(v2)] = p1[f]
                        p2_int[f + " | " + self.cond_feature+"="+str(v1)+","+str(v2)] = p2[f]
                    else:
                        p1_int[f + " | " + self.cond_feature+"="+str(v1)+","+str(v2)] = 0
                        p2_int[f + " | " + self.cond_feature+"="+str(v1)+","+str(v2)] = 0


            new_cols.append(self.cond_feature)
            p1_int[self.cond_feature] = p1[self.cond_feature]
            p2_int[self.cond_feature] = p2[self.cond_feature]

            self.int_cols = list(new_cols)
            p1, p2 = p1_int, p2_int





        return p1, p2

    def fit(self, X):
        self.X_ = X
        _, _ = self.get_pair(self.X_.iloc[0]) 

        fvals, feat_to_indices, feat_to_fvals = [], {}, {}
        if self.interactions:
            cols_to_use = self.int_cols
        else:
            cols_to_use = self.cols

        for f in cols_to_use:
            start = len(fvals)
            vals = self.feature_values[f.split(' | ')[0]]
            val_pairs = itertools.combinations(vals, 2)

            fvals += [f+"_"+str(v1)+","+str(v2) for (v1, v2) in val_pairs]
            feat_to_fvals[f] = list(val_pairs)
            feat_to_indices[f] = list(range(start, len(fvals)))


        pairs, true_labels = [], []
        for row in self.X_.iterrows():
            row = row[1]
            p1, p2 = self.get_pair(row)
            pairs.append((p1, p2))
            true_labels.append(row["chosen"])

        self.pairs_ = pairs
        self.true_labels_ = true_labels

        def error(rule_vars):

            rules = {f: np.array(rule_vars[feat_to_indices[f]]) for f in cols_to_use}
            # rules = {f: [np.round(v) for v in rules[f]] for f in cols_to_use}
            rules = [PairwiseHeuristicRule(feat, self.feature_values[feat.split(' | ')[0]], rules[feat], self.feature_signs[feat.split(' | ')[0]]) for feat in cols_to_use]

            preds, labels = [], []
            indices = list(range(len(self.pairs_)))
            random.shuffle(indices)
            # indices = indices[:30]

            for idx in indices:
                p1, p2 = self.pairs_[idx]
                pred = self._predict_single((p1, p2), rules)
                preds.append(pred)
                labels.append(self.true_labels_[idx])

            preds = np.array(preds)
            labels = np.array(labels)

            err = -np.mean(labels*np.log(preds) + (1-labels)*np.log(1-preds))

            return err + self.lambda1*np.linalg.norm(rule_vars, ord=1)/len(rule_vars) + self.lambda2*np.linalg.norm(rule_vars, ord=2)/len(rule_vars)

        rule_vars_init = [0]*len(fvals)

        cons_inds = []
        for f in cols_to_use:
            indices = feat_to_indices[f]
            val_pairs = feat_to_fvals[f]
            for j in range(len(val_pairs)):
                for k in range(j+1, len(val_pairs)):
                    pair1 = val_pairs[j]
                    pair2 = val_pairs[k]
                    if pair1[0] >= pair2[0] and pair1[1] == pair2[1]:
                        cons_inds.append((indices[j], indices[k]))

                    if pair1[0] == pair2[0] and pair1[1] <= pair2[1]:
                        cons_inds.append((indices[j], indices[k]))

            # cons_inds += [(i, i-1) for i in indices[1:]]
        
        ineq_cons = {'type': 'ineq', 'fun': lambda u: np.array([u[i] - u[j] for i, j in cons_inds])}

        bounds = Bounds([0]*len(fvals), [10]*len(fvals))
        # bounds = Bounds([0]*len(fvals))
        res = minimize(error, rule_vars_init, method='SLSQP', constraints=ineq_cons, options={'maxiter': 100, 'disp': False}, )

        rule_vars = res.x
        rules = {f: np.array(rule_vars[feat_to_indices[f]]) for f in cols_to_use}
        rules = {f: [np.round(v) for v in rules[f]] for f in cols_to_use}
        rules = [PairwiseHeuristicRule(feat, self.feature_values[feat.split(' | ')[0]], rules[feat], self.feature_signs[feat.split(' | ')[0]]) for feat in cols_to_use]
        self.ir = list(rules)

        for r in rules:
            if r.__str__() != "":
                print (r)

        return self
    
    def predict(self, X, rules):
        predictions = [self._predict_single(x, rules) for x in X]
        return predictions

    def _predict_single(self, x, rules, discrete=False):

        p1, p2 = x
        if not self.interactions:
            rule_apps = [rules[i].apply_pair(p1[f], p2[f], discrete) for i, f in enumerate(self.cols)]
        else:        
            rule_apps = [rules[i].apply_pair(p1[f], p2[f], discrete) for i, f in enumerate(self.int_cols)]

        pred = np.mean(rule_apps)
        # print (x, rule_apps)

        return pred

    def score(self, X, y=[]):
        preds, true_labels = [], []
        for row in X.iterrows():
            row = row[1]

            p1, p2 = self.get_pair(row)
            pred = self._predict_single((p1, p2), self.ir, discrete=True)
            pred = int(pred > 0)
            preds.append(pred)

            true_labels.append(row["chosen"])
        
        preds = np.array(preds)
        true_labels = np.array(true_labels)
        # print (list(zip(preds, true_labels)))
        acc = np.mean(preds == true_labels)
        return acc       


def sigmoid_sgn(x, temp=1):
    return sigmoid(x*temp)

