"""
classify.py
"""
# coding: utf-8
from collections import Counter, defaultdict
from itertools import chain, combinations
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import pickle
import json
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from nltk.corpus import stopwords
stop_word = set(stopwords.words('english'))

def get_tweet_data(filename):
    with open(filename,'r') as f:
        tweets = json.load(f)
    return tweets

def filter_tweets(tweets):
    for tweet in tweets:
        #Get rid of links
        tweet = re.sub('http\S+',' ',tweet)
        #Get rid of mentions
        tweet = re.sub('@\S+',' ',tweet)
    return tweet


def tokenize(tweet, keep_internal_punct=True, collapse_url = True, collapse_mention = True):
    tweet = tweet.lower()
    tweet = re.sub(u"(\u2018|\u2019)","'",tweet)
    tweet = re.sub('#','',tweet)
    if collapse_url:
        tweet = re.sub('http\S+','THIS_IS_URL',tweet)
    if collapse_mention:
        tweet = re.sub('@\S+','THIS_IS_MENTION',tweet)
    if not keep_internal_punct:
        tweet = re.sub('\W+',' ',tweet).split()
    else:
        tweet = re.findall('\w[^\s]*\w|\w+', tweet)
    return np.array(tweet)

#Shuffle to get random sample of data
def shuffle_two_list(list1,list2):
    c = list(zip(list1,list2))
    random.shuffle(c)
    list1, list2 = zip(*c)
    return list1,list2

def assign_label_to_tweet(male_tweets,female_tweets):
    tweets = []
    labels = []
    for tweet in male_tweets:
        tweets.append(tweet)
        labels.append(1)
    for tweet in female_tweets:
        tweets.append(tweet)
        labels.append(0)
    tweets, labels = shuffle_two_list(tweets,labels)
    return tweets, np.array(labels)

def get_train_test(tweets,labels):
    stratSplit = StratifiedShuffleSplit(n_splits = 3, test_size = 0.2,random_state = 42)
    for train_index, test_index in stratSplit.split(tweets,labels):
        X_train, X_test = tweets[train_index], tweets[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
    return X_train,X_test, y_train, y_test

def token_features(tokens, feats):
    for token in tokens:
        if token not in stop_word:
            feats['token='+token]+=1


def token_pair_features(tokens, feats, k=5):
    token_pair = []
    for i in range(len(tokens)):
        token_nearby = tokens[i:(i+k)]
        if len(token_nearby) == k:
            temp = list(combinations(token_nearby,2))
            for token1, token2 in temp:
                if token1 not in stop_word or token2 not in stop_word:
                    #token_pair is now a list of tuple
                    token_pair.append((token1,token2))
        elif len(token_nearby) < k:
            break
            
    for token1, token2 in token_pair:
        feats['token_pair='+token1.lower()+'__'+token2.lower()] += 1

neg_words = ['bad', 'hate', 'horrible', 'worst', 'boring']
pos_words = ['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful']
def lexicon_features(tokens, feats):
    #Initialize so (neg_words,0) or (pos_words,0) also appear in feats
    feats['pos_words'] = 0
    feats['neg_words'] = 0
    for token in tokens:
        if token.lower() in pos_words:
            feats['pos_words']+=1
        if token.lower() in neg_words:
            feats['neg_words'] +=1

def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for func in feature_fns:
        func(tokens,feats)
    return sorted(feats.items(),key = lambda x: x[0])


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """CREATING VOCAB AND CONSTRICTING FEATURES TO HAVE AT LEAST MIN_FREQ APPEARANCE"""
    #List of list, first list corresponds to the features of doc 1, second to doc 2, so on
    feats = []
    #To get all the possible terms for the vocab dict and able to universally sort them
    feats_list = []
    temp_list = []
    for tokens in tokens_list:
        features = featurize(tokens,feature_fns)
        #each of the sublist corresponds to the feature of a document
        feats.append(features)
        feats_list.extend(features)
    feats_list = sorted(feats_list,key = lambda x: x[0])
    
    #If vocab is None, then start filling the vocab dict
    if not vocab:
        vocab = defaultdict(lambda : len(vocab))

        for tup in feats_list:
            temp_list.append(tup[0])
        #count contains term : number of doc it appears in
        count = Counter(temp_list)
        for tup in feats_list:
            if count[tup[0]] >= min_freq:
                #looking up a term, defaultdict will assign an incremental value to the term for us
                vocab[tup[0]]
    
    """CREATING CSR"""
    X = np.zeros((len(tokens_list),len(vocab)))
    for i, tokens in enumerate(feats):
        for token in tokens:
            #If the term is in vocab, which means it appears in at least min_freq docs
            if token[0] in vocab:
                #get the column where the token belongs
                j = vocab[token[0]]
                X[i,j] = token[1]
    #X = csr_matrix(X)
    return X, vocab
    
    

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(n_splits = k)
    accuracies = []
    for train_index, test_index in cv.split(X):
        clf.fit(X[train_index],labels[train_index])
        predicted = clf.predict(X[test_index])
        accuracies.append(accuracy_score(labels[test_index],predicted))
    return np.mean(accuracies)
                    


def eval_all_combinations(docs, labels, punct_vals, collapse_urls, collapse_mentions,
                          feature_fns, min_freqs):
    result = []
    for value in punct_vals:
        for value1 in collapse_urls:
            for value2 in collapse_mentions:
                tokens_list = [tokenize(d,keep_internal_punct = value, collapse_url = value1, collapse_mention = value2) for d in docs]
                for min_freq in min_freqs:
                    for i in range(1,len(feature_fns)+1):
                        #Feature lists is a list of tuple for each value of i
                        feature_lists = combinations(feature_fns,i)
                        for feature_list in feature_lists:
                            feature_list = list(feature_list)
                            X, vocab = vectorize(tokens_list,feature_list,min_freq)
                            avg = cross_validation_accuracy(MultinomialNB(),
                                                            X,labels,5)
                            result.append({
                                    'features':tuple(feature_list),
                                    'punct':value,
                                    'url':value1,
                                    'mention':value2,
                                    'accuracy':avg,
                                    'min_freq':min_freq
                                })
    result = sorted(result,key = lambda x: (x['accuracy'],x['min_freq']), reverse = True)
    return result
            

def mean_accuracy_per_setting(results):
    mean_acc = []
    setting_to_acc = defaultdict(list)
    for result in results:
        acc = result['accuracy']
        if result['punct'] == True:
            setting_to_acc['punct=True'].append(acc)
        if not result['punct']:
            setting_to_acc['punct=False'].append(acc)
        setting_to_acc['min_freq='+str(result['min_freq'])].append(acc)
        setting_to_acc['collapse_url='+str(result['url'])].append(acc)
        setting_to_acc['collapse_mention='+str(result['mention'])].append(acc)
        string = 'features='
        for feature in result['features']:
            string += feature.__name__ + ' '
        setting_to_acc[string].append(acc)
            
    for setting, accuracies in setting_to_acc.items():
        mean_acc.append((np.mean(accuracies),setting))
    
    return sorted(mean_acc,key = lambda x: x[0], reverse = True)
    
            

def fit_best_classifier(docs, labels, best_result,best_params):
    tokens_list = [tokenize(d,keep_internal_punct = best_result['punct'],collapse_url = best_result['url'],
                           collapse_mention = best_result['mention']) for d in docs]
    X, vocab = vectorize(tokens_list,list(best_result['features']),best_result['min_freq'])
    clf = MultinomialNB(fit_prior = best_params['fit_prior'],alpha = best_params['alpha'])
    clf.fit(X,labels)
    return clf, vocab,clf.predict(X)

def predict_test_data(clf,docs,best_result,vocab):
    tokens_list = [tokenize(d,keep_internal_punct = best_result['punct'],collapse_url = best_result['url'],
                           collapse_mention = best_result['mention']) for d in docs]
    X, vocab = vectorize(tokens_list,list(best_result['features']),best_result['min_freq'],vocab)
    return clf.predict(X)

def print_top_positive(tweets,X_test,clf,n):
    prediction = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    probs = [prob[1] for prob in probs]
    top = []
    for tweet, probability in zip(tweets,probs):
        top.append({
                'tweet': tweet,
                'prediction': 1,
                'proba': round(probability,4)
                })
    top = sorted(top,key = lambda x: x['proba'],reverse = True)
    for i in range(n):
        info = top[i]
        print('Prediction = {} probability = {}'.format(1,info['proba'])+'\t\t\t'+info['tweet'])

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    predicted = clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    incorrect = []
    for doc, label, prediction, probability in zip(test_docs,test_labels,predicted,prob):
        if label != prediction:
            if prediction == 0:
                incorrect.append({
                        'doc':doc,
                        'truth':label,
                        'predicted':prediction,
                        'proba': round(probability[0],6)
                    })
            elif prediction == 1:
                incorrect.append({
                        'doc':doc,
                        'truth':label,
                        'predicted':prediction,
                        'proba': round(probability[1],6)
                    })
    incorrect = sorted(incorrect, key = lambda x: x['proba'], reverse = True)
    for i in range(n):
        info = incorrect[i]
        print('\n')
        print('truth={} predicted={} proba={}'.format(info['truth'],info['predicted'],info['proba']))
        print(info['doc'])

def top_coefs(clf, label, n, vocab):
    vocab = sorted(vocab.items(), key = lambda x: x[1])
    term_vocab = np.array([tup[0] for tup in vocab])
    if label == 0:
        coef = clf.coef_[0]
        top_idx = np.argsort(coef)[:n]
        top_coef = coef[top_idx]
    else:
        coef = clf.coef_[0]
        top_idx = np.argsort(coef)[::-1][:n]
        top_coef = coef[top_idx]
        
    
    top_term = term_vocab[top_idx]
    return [x for x in zip(top_term,top_coef)]

def fine_tune_model(tweets, labels,best_result):
    param_grid = {'alpha':list(np.arange(0.0,5.0,0.05)),'fit_prior':[True,False]}
    tokens_list = [tokenize(tweet,keep_internal_punct = best_result['punct'],
                           collapse_url = best_result['url'],collapse_mention = best_result['mention']) 
                   for tweet in tweets]
    feature_list = list(best_result['features'])
    X, vocab = vectorize(tokens_list,feature_list,best_result['min_freq'])
    clf = MultinomialNB()
    ran_search = RandomizedSearchCV(clf,param_grid,cv = 5, scoring = 'accuracy')
    ran_search.fit(X,labels)
    return ran_search.best_params_

def save_file(filename,info):
	with open(filename,'w') as f:
		json.dump(info,f)

def main():
    print('Reading and labeling raw tweets...')
    male_tweets = list(set(get_tweet_data('maletweets.txt')))
    female_tweets = list(set(get_tweet_data('femaletweets.txt')))
    tweets, labels = assign_label_to_tweet(male_tweets,female_tweets)
    save_file('totaltweets.txt',tweets)
    info = {}
    print('\nSampling train/test set using Stratified Sampling...')
    X_train,X_test,y_train,y_test = get_train_test(np.array(tweets),labels)
    info['test_set'] = len(X_test)
    print('\nEvaluating over multiple preprocessing settings...')
    feature_fns = [token_features,token_pair_features]
    results = eval_all_combinations(X_train,y_train,
                                    [True,False],
                                    [True, False],
                                    [True,False],
                                    feature_fns,
                                    list(range(1,10)))
    best_result = results[0]
    worst_result = results[-1]
    #Printing
    print('\nBest cross-validation result:\n{}'.format(str(best_result)))
    print('Worst cross-validation result:\n{}'.format(str(worst_result)))
	
    print('\nFine-tuning Multinomial Naive Bayes...')
    best_params = fine_tune_model(X_train,y_train,best_result)
    clf, vocab, train_prediction = fit_best_classifier(X_train, y_train, best_result,best_params)
    print('Accuracy of fine-tuned MNB on train set: {}'.format(str(accuracy_score(y_train,train_prediction))))
    #<!!!!> HAVE TO PASS IN VOCAB, to ensure consistency
    test_prediction = predict_test_data(clf, X_test, best_result,vocab)
    accuracy = accuracy_score(y_test,test_prediction)
    print('Accuracy of fine-tuned MNB on test set: {}'.format(str(accuracy)))
    info['accuracy'] = accuracy
    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('Top words used by female critic:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 10, vocab)]))
    print('\Top words used by male critic')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 10, vocab)]))
    
	#For summarize.py
    class0 = []
    class1 = []
    example0 = []
    example1 = []
    for i in range(len(test_prediction)):
	    if test_prediction[i] == 0:
		    class0.append(int(test_prediction[i]))
		    example0.append(X_test[i])
	    if test_prediction[i] == 1:
		    class1.append(int(test_prediction[i]))
		    example1.append(X_test[i])
			
    info['class_0'] = class0
    info['class_1'] = class1
    info['example_0'] = example0[0]
    info['example_1'] = example1[0]
    save_file('classinfo.txt',info)
	
if __name__ == '__main__':
    main()