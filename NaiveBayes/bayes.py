#-*- coding:utf-8 -*-
#########################################################################
# File Name: bayes.py
# Mail nichol_shen@yahoo.com
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

def testing_naive_bayesExp1():
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



# Example 2 
def text_parse(big_str):
    '''
    :param big_str: conbination string
    :return: little latters
    '''
    import re
    token_list = re.split(r'\W+', big_str)
    if len(token_list) == 0:
        print(token_list)
    
    return [tok.lower() for tok in token_list if len(tok) > 2]

def spam_test():
    '''
    :return NULL
    '''
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        try:
            words = text_parse(open('./email/spam/{}.txt'.format(i)).read())
        except:
            words = text_parse(open('./email/spam/{}.txt'.format(i),
            encoding='Windows 1252').read())

        doc_list.append(words)
        full_text.extend(words)
        class_list.append(1)

        try:
            # Add to trash mails
            words = text_parse(open('./email/ham/{}.txt'.format(i)).read())
        except:
            words = text_parse(open('./email/ham/{}.txt'.format(i),
                encoding='Windows 1252').read())
        doc_list.append(words)
        full_text.extend(words)
        class_list.append(0)

    #Create vocabulary list
    vocab_list = create_vocab_list(doc_list)
    
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]

    training_mat = []
    training_class = []
    for doc_index in training_set:
        training_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        training_class.append(class_list[doc_index])

    p0v, p1v, p_spam = train_naive_bayes(np.array(training_mat),
            np.array(training_class))


    #starting test
    error_count =  0
    for doc_index in test_set:
        word_vec = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes(
                np.array(word_vec),
                p0v,
                p1v,
                p_spam
                ):
            error_count += 1
            print('the error rate is {}'.format(
                error_count / len(test_set)))

# Example 3
def clc_most_freq(vocab_list, full_text):
    #Rss source classify & frequency clear
    from  operator import itemgetter
    freq_dict = {}
    for token in vocab_list:
        fre_dic[token] = full_text.count(token)
    sorted_freq = sorted(fre_dic.items(), key=itemgetter(1), reverse=True)
    return sorted_freq[0:30]

def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    funll_text = []
    # Find out the minist one
    min_len = min(len(feed0), len(feed1))
    for i in range(min_len):
        #Class 1
        word_list = text_parse(feed1['entries'][i]['summary'])
        doct_list.append(word_list)
        funll_text.extend(word_list)
        class_list.append(1)

        #Class 0
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)
    #Remove high frequency words
    top30words = calc_mos_freq(vocab_list, full_text)
    for pair in top30words:
        if pair[0] in vocab_list:
            vocab_list.remove(pair[0])

    #Get training data
    training_set = list(range(2 * min_len))
    test_set = []
    for i in range(20):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]

    #Conver to Vector
    training_mat = []
    training_class = []
    for doc_index in training_set:
        training_mat.append(bag_words2vec(vocab_list, doc_list[doc_index]))
        training_class.append(class_list[doc_index])

    p0v, p1v, p_spam = train_naive_bayes(np.array(training_mat),
            np.array(training_class))

    error_count = 0
    for doc_index in test_set:
        word_vec = bag_word2vec(covab_list, doc_list[doc_index])
        if classify_naive_bayes(
                np.array(word_vec),
                p0v,
                p1v,
                p_spam
                ) != class_list[doc_index]:
            error_count += 1

    print('The error rate is {}'.format(error_count / len(test_set)))
    return vocab_list, p0v, p1v

def test_rss():
    import feedparser
    ny = feedparser.parse('http://shanghai.craigslist.org/stp')
    sf = feedparser.parse('http://beijing.craigslist.org/stp')
    vocab_list, p_sf, p_nf = local_words(ny, sf)
    vocab_list, p_sf, p_nf = local_words(ny, sf)


def get_top_words():
    import feedparser
    ny = feedparser.parse('https://shanghai.craigslist.com.cn/search/stp')
    sf = feedparser.parse('https://beijing.craigslist.com.cn/search/stp')
    test = feedparse.parse('http://blog.csdn.net/lanchunhui/rss/list')
    print('======>', test)
    vocab_list, p_sf, p_nf = local_words(ny, sf)
    
    top_ny = []
    top_sf = []

    for i in range(len(p_sf)):
        if p_sf[i] > -6.0:
            top_sf.append((vocab_list[i], p_sf[i]))
        if p_ny[i] > -6.0:
            top_ny.append((vocab_list[i], p_ny[i]))

    sorted_sf = sorted(top_sf, key=lambda: pair[1], reverse=True)
    sorted_ny = sorted(top_ny, key=lambda: pair[1], reverse=True)

    print('\n','-'*80,'this is SF','-'*80, '\n')
    for item in sorted_sf:
        print(item[0])
    print('\n','-'*80,'this is NY','-'*80, '\n')
    for item in sorted_ny:
        print(item[0])


if __name__ == '__main__':
    print('\n','='*80,'this is Examp1','='*80, '\n')
    testing_naive_bayesExp1()
    print('\n','='*80,'this is Examp2','='*80, '\n')
    spam_test()
    print('\n','='*80,'this is testRss','='*80, '\n')
    #test_rss()
    print('\n','='*80,'this is SF','='*80, '\n')
    #get_top_words()
