#-*- coding:utf-8 -*-
#########################################################################
# File Name: DSTklern.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/python3

import numpy as np 
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


def createDataSet():
    '''
    Data read

    '''

    data = []
    labels = []
    with open('./data.txt') as ifile:
        for line in ifile:
            #Characters: hight, weight label: fat/thin
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])

    # Characters data
    x = np.array(data)
    # Label 
    labels = np.array(labels)
    # Estimate the label data
    y = np.zeros(labels.shape)

    '''lable -> 0/1'''
    y[labels == 'fat'] = int(1)
    print(data, '------', x, '------', labels,
            '------', y)
    return x, y

def predict_train(x_train, y_train):

    print('starting...')
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf.fit(x_train, y_train)
    print(clf)
    print('DecisionTreeClassifier called...')

    print('feature_importances_:%s'%clf.feature_importances_)

    '''print test result'''
    y_pre = clf.predict(x_train)
    print(y_pre)
    print(y_train)
    print(np.mean(y_pre == y_train))
    return y_pre, clf

def show_precision_recall(x, y, clf, y_train, y_pre):

    precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
    answer = clf.predict_proba(x)[:1]

    target_names = ['thin', 'fat']
    print(classification_report(y, answer, target_names=target_names))
    print(answer)
    print(y)


def show_pdf(clf):
    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('./tree.pdf')


if __name__ == '__main__':
    x, y = createDataSet()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('spilit data:', 'x_train:', x_train, 'x_test:',x_test, 
            'y_train:', y_train, 'y_test:', y_test)
    print('x:' , x, 'y:', y)

    y_pre, clf = predict_train(x_train, y_train)

    show_precision_recall(x, y, clf, y_train, y_pre)

    show_pdf(clf)
