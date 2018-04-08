#-*- coding:utf-8 -*-
#########################################################################
# File Name: bayes.py
# Author: Shen Bo
# Created Time: Sun 08 Apr 2018 09:27:57 PM DST
#########################################################################
#!usr/bin/env python3

"""
Bayes equation

p(xy) = p(x|y) * p(y) = p(y|x) * p(x)
p(x|y) = p(y|x) * p(x)/p(y)
"""
import numpy as np

# Example 1 Shelding insult comments

def load_data_set():
    '''
    Fake data set creation
    return: word list posting_list, Classes class_vec
    '''

    posting_list = [
           ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage', 'fuck', 'off'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    class_vec = [0, 1, 0, 1, 0, 1] # 1 is inslting words, 0 not

    return posting_list, class_vec

def create_vocab_list(data_set):
    '''
    get all vovabulary set
    param: data_set
    return: all vocabulary set(not repeat words)
    '''

    vocab_set = set()
    for item in data_set:
        # Unite 2 sets
        vocab_set = vocab_set | set(item)

    print('===vocab_set: ', vocab_set)
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    '''
    Retrvase & confirm words' apparence, if appared assign 1
    param: vocab_list
    param: input_set
    return: match_list [0, 1, 1, 0]
    '''

    result = [0] * len(vocab_list)

    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            print('the word: {} is not in my vocabulary list'.format(word))
    
    print('result: ', result)
    return result

def _train_naive_bayes(train_mat, train_category):
    '''
    Naive bayes demo
    param: train_mat: type is ndarray
            example [[0, 1, 0, 1], [], []]
    param: train_category: the associated categories,
            example [0, 1, 0]
    reutn
    '''

    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])

    pos_abusive = np.sum(train_category) / train_doc_num

    p0num = np.zeros(words_num)
    p1num = np.zeros(words_num)

    # Times of data sets' words
    p0num_all = 0
    p1num_all = 0

    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])

    p1vec = p1num / p1num_all
    p0vec = p0num / p0num_all

    return p0vec, p1vec, pos_abusive


def train_naive_bayes(train_mat, train_category):
    '''
    Naive bayes demo
    param: train_mat: type is ndarray
            example [[0, 1, 0, 1], [], []]
    param: train_category: the associated categories,
            example [0, 1, 0]
    reutn
    '''

    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])

    pos_abusive = np.sum(train_category) / train_doc_num

    p0num = np.zeros(words_num)
    p1num = np.zeros(words_num)

    # Times of data sets' words
    p0num_all = 2.0
    p1num_all = 2.0

    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])

    p1vec = p1num / p1num_all
    p0vec = p0num / p0num_all

    return p0vec, p1vec, pos_abusive

def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):
    '''
    Algorithm:
        Covert Times to plus
        Times: P(C|F1F2...Fn) = P(F1F2...Fn|C)/P(F1F2...Fn)
    Adds:  P(F1|C)*p(F2|C) ...P(Fn|C)P(C) = log(P(F1|C)) + log(P(F2|C)) +
        ... + log(P(C))
    param: vec2classify: data ready to masure
    param: p0vec: Category0, normal doc[log(P(F1|C0)), log(P(F2|C0)), ...]
    param: p1vec: Category1, abnormal doc[log(P(F2|C1)), log(P(F2|C1)),...]
    param: p_class1: Category1, Insulting words
    return: Category 1 or 0
    '''

    p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class1)

    if p1 > p0:
        return 1
    else:
        return 0

def bag_words2vec(vocab_list, input_set):
    #Compare with the origin one
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
        else:
            print('the word: {} is not in my vocabluary'.format(word))
    
    return result

def testing_naive_bayes():
    '''
    test Naive bayes algorithm
    return: no return
    '''

    # Load data sets
    list_post, list_classes = load_data_set()
    # Create vocablulary sets
    vocab_list  = create_vocab_list(list_post)

    # Calculate words apparence
    train_mat = []
    for post_in in list_post:
        train_mat.append(
                # return m * len(vocab_list) matrix, contents filled with 0 or 1
                set_of_words2vec(vocab_list, post_in)
        )

    # Training data
    p0v, p1v, p_abusive = train_naive_bayes(np.array(train_mat),
            np.array(list_classes))

    print(p0v, p1v, p_abusive)

    # Testing data
    test_one  = ['love', 'my', 'dalmation']
    test_one_doc = np.array(set_of_words2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc,
        p0v, p1v, p_abusive)))

    test_two = ['stupid', 'garbage', 'fuck', 'off']
    test_two_doc = np.array(set_of_words2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc,
        p0v, p1v, p_abusive)))


if __name__ == '__main__':
    testing_naive_bayes()
