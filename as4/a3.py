# coding: utf-8

#
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
unique_token.extend(list(set(token_list)))unique_token.extend(list(set(token_list)))unique_token.extend(list(set(token_list)))
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    token = []
    for genres in movies['genres']:
        token.append(tokenize_string(genres))
    movies = movies.assign(tokens=pd.Series(token,movies.index))
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    #remove duplicate??
    tf_list = []
    #Contain unique term for each movie, Ex: ['horror','horror'] => ['horror']
    unique_token = []
    #Get all the tf(i,d) of each term i for each document d
    for token_list in movies['tokens']:
        tf = defaultdict(lambda: 0)
        for token in token_list:
            tf[token] += 1
        tf_list.append(tf)
        unique_token.extend(list(set(token_list)))
        
    #Get all df(i) for each term i
    df = Counter(unique_token)
    unique_token = sorted(set(unique_token))
    vocab = defaultdict(lambda:len(vocab))
    for token in unique_token:
        vocab[token]
    
    #Constructing csr matrix for each row
    N = movies.shape[0]
    matrix_list = []
    for i in range(N):
        tf = tf_list[i]
        X = np.zeros((1,len(vocab)))
        max_k = max(tf.values())
        #Insert each term and its value into the matrix in sorted order
        terms = sorted(tf.keys())
        for term in vocab:
            if term in terms:
                j = vocab[term]
                X[0,j] = tf[term]/max_k * np.log10(N/df[term])
                
        matrix_list.append(csr_matrix(X))
    
    movies = movies.assign(features=pd.Series(matrix_list,movies.index))
    
    return movies, dict(vocab)
        


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
	#Convert back to np array for faster dot product
    a = a.toarray()[0]
    b = b.toarray()[0]
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ratings = []
    for user, movie in zip(ratings_test['userId'],ratings_test['movieId']):
        ratings_per_user = ratings_train[ratings_train.userId == user]
        csr_1 = movies.loc[movies['movieId']==movie, 'features'].iloc[0]
        ratings_per_user = ratings_per_user.join(movies.set_index('movieId'),on = 'movieId',how = 'inner')
        weighted_total = 0
        sum_cos = 0
        for movieId, rating, csr_matrix_feature in zip(ratings_per_user['movieId'],ratings_per_user['rating'],ratings_per_user['features']):
            cos_sim = cosine_sim(csr_matrix_feature,csr_1)
            if cos_sim > 0:
                weighted_total += cos_sim * rating
                sum_cos+=cos_sim
        if sum_cos > 0:
            ratings.append(weighted_total/sum_cos)
        elif sum_cos == 0:
            ratings.append(ratings_per_user['rating'].mean())
    return np.array(ratings)
            

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
