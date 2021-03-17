
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.model_selection import train_test_split

import numpy as np
from collections import defaultdict
import logging

def evaluate_splited(model,X_train,Y_train,X_test,Y_test, alg = None, classifier="lr",fast=True,cv=10,normalize=False,standarlize=True,random_state = None,return_y = False):
    X_train = [model.paper_embeddings[model.paper2id[paper]] for paper in X_train ]
    X_test  = [model.paper_embeddings[model.paper2id[paper]] for paper in X_test  ]

    if normalize:
        X_train = sk_normalize(X_train)
        X_test  = sk_normalize(X_test)
    if standarlize:
        scaler = StandardScaler()
        X_trian = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    clf = LogisticRegression()
    df = defaultdict(list)
    micros = []
    macros = []
    for i in range(cv):
        clf = LogisticRegression()
        if classifier.lower() == "svm":
            clf = SVC(cache_size=5000)
        elif classifier.lower() == "mlp":
            clf = MLPClassifier()
        elif classifier.lower() == "gnb":
            clf = GaussianNB()
        elif classifier.lower() == "mnb":
            clf = MultinomialNB()

        clf.fit(X_train,Y_train)
        prediction = clf.predict(X_test)
        micro = f1_score(Y_test, prediction, average='micro')
        macro = f1_score(Y_test, prediction, average='macro')
        micros.append(micro)
        macros.append(macro)

    micros = np.mean(micros)
    macros = np.mean(macros)


    df["micro"].append(np.mean(micro))
    df["macro"].append(np.mean(macro))
    #df["alg"].append(alg)
    #df["data"].append(str(data))
    #df["total_samples"] = model.total_samples
    #df["negative"].append(model.negative)
    #df["walk_window"].append(model.walk_window)
    #df["walk_probability"].append(model.walk_probability)
    #df["L2"].append(model.l2)
    logging.info("f1_micro %.4f, f1_macro %.4f" % (micros,macros))

    if fast:
        if return_y:
            return micros,macros,Y_test,prediction
        return micros,macros
    else:
        return pd.DataFrame(df)



def evaluate(model, data, alg = None, classifier="lr",fast=True,ratio = None,cv=10,normalize=False,standarlize=True,random_state = None,return_y = False):
    X = []
    Y = []

    if hasattr(data,'labels'):
        labels = data.labels
    elif isinstance(data,dict):
        labels = data
    elif isinstance(data,defaultdict):
        labels = data


    for y,key in enumerate(labels.keys()):
        for index,paper in enumerate(labels[key]):
            if paper not in model.paper2id:
                continue
            X.append(model.paper_embeddings[model.paper2id[paper]])
            Y.append(y)

    if normalize:
        X = sk_normalize(X)
    if standarlize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    clf = LogisticRegression()
    df = defaultdict(list)
    if ratio is None:
        ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for r in ratio:
        if r <= 0:
            continue
        elif r >= 1:
            break

        micros = []
        macros = []
        for i in range(cv):
            clf = LogisticRegression()
            if classifier.lower() == "svm":
                clf = SVC(cache_size=5000)
            elif classifier.lower() == "mlp":
                clf = MLPClassifier()
            elif classifier.lower() == "gnb":
                clf = GaussianNB()
            elif classifier.lower() == "mnb":
                clf = MultinomialNB()

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-r,random_state=random_state)
            clf.fit(X_train,Y_train)
            prediction = clf.predict(X_test)
            micro = f1_score(Y_test, prediction, average='micro')
            macro = f1_score(Y_test, prediction, average='macro')
            micros.append(micro)
            macros.append(macro)

        micros = np.mean(micros)
        macros = np.mean(macros)


        df["ratio"].append(r)
        df["micro"].append(np.mean(micro))
        df["macro"].append(np.mean(macro))
        #df["alg"].append(alg)
        #df["data"].append(str(data))
        #df["total_samples"] = model.total_samples
        #df["negative"].append(model.negative)
        #df["walk_window"].append(model.walk_window)
        #df["walk_probability"].append(model.walk_probability)
        #df["L2"].append(model.l2)
        logging.info("ratio: %.4f : f1_micro %.4f, f1_macro %.4f" % (r,micros,macros))

    if fast:
        if return_y:
            return micros,macros,Y_test,prediction
        return micros,macros
    else:
        return pd.DataFrame(df)

def evaluate_multilabel(model, data, alg = None, classifier="lr",fast=False,ratio = None, cv = 10, random_state = None,normalize=False):

    X = []
    Y = []

    papers = []

    if hasattr(data,'labels'):
        labels = data.labels
    elif isinstance(data,dict):
        labels = data
    elif isinstance(data,defaultdict):
        labels = data

    for v in labels.values():
        papers += list(v)
    papers = set(papers)

    paper2id = {}
    for pid,paper in enumerate(papers):
        paper2id[paper] = pid
        X.append(model.paper_embeddings[model.paper2id[paper]])

    Y = np.zeros((len(X),len(labels)))

    for y,key in enumerate(labels.keys()):
        for index,paper in enumerate(labels[key]):
            pid = paper2id[paper]
            Y[pid][y] = 1

    if normalize:
        X = sk_normalize(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = defaultdict(list)
    if ratio is None and cv == 1:
        ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for r in ratio:
        if r <= 0:
            continue
        elif r >= 1:
            break

        if classifier.lower() == 'lr':
            clf = LogisticRegression()
        elif classifier.lower() == "svm":
            clf = SVC(cache_size=5000)
        elif classifier.lower() == "mlp":
            clf = MLPClassifier()
        elif classifier.lower() == "gnb":
            clf = GaussianNB()
        elif classifier.lower() == "mnb":
            clf = MultinomialNB()

        micros = []
        macros = []
        for i in range(cv):
            micro,macro = evaluateNodeClassification(X,Y,1-r,clf=clf,random_state = random_state)
            micros.append(micro)
            macros.append(macro)
        micros = np.mean(micros)
        macros = np.mean(macros)

        df["ratio"].append(r)
        df["micro"].append(micros)
        df["macro"].append(macros)
        #df["alg"].append(alg)
        #df["data"].append(str(data))
        #df["total_samples"].append(model.total_samples)
        #df["negative"].append(model.negative)
        #df["walk_window"].append(model.walk_window)
        #df["walk_probability"].append(model.walk_probability)
        #df["L2"].append(model.l2)

        logging.info("ratio: %.4f : f1_micro %.4f, f1_macro %.4f" % (r,micros,macros))


    if fast:
        return micros,macros
    else:
        return df

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction

def evaluateNodeClassification(X, Y, test_ratio,clf=None,random_state = None):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio,random_state = random_state)
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    if clf == None:
        classif2 = TopKRanker(LogisticRegression())
    else:
        classif2 = TopKRanker(clf)

    classif2.fit(X_train, Y_train)
    prediction = classif2.predict(X_test, top_k_list)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    return (micro, macro)
