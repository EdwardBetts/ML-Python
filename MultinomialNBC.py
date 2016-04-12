"""
File: MultinomialNBC.py
-----------------------
This program implements multinomial Naive Bayes for a sample of Spam and
Ham from the LingSpam 2010 publication and database.  This uses the
bare subset of data, without prior lemmatization or stop-word removal.
"""

import os
import sys
import operator
import math
import string
import pickle
from operator import itemgetter
from cStringIO import StringIO
from itertools import ifilterfalse
from random import shuffle

def word_iter_stringio(counter, filename, folder, fileText, vocabDict):
# Opens file and adds the unique words to both dictionary and respective document
    pathname = folder + filename
    with open(pathname) as f:
        io = StringIO(f.read())
        fileText.append('')
        for line in io:
            for word in line.split():
                if word.isalnum() and not is_number(word):
                    add_to_dict(word, vocabDict)
                    fileText[counter] += word + " "
        #fileText[counter] = unique_words(fileText[counter])
        f.close()

def is_number(s):
# Is it a number? (pure numbers are ignored for this NBC model)
    try:
        float(s)
        return True
    except ValueError:
        return False

def add_to_dict(word, vocabDict):
# Includes word count, for prioritizing dictionary
    word =     word.lower()
    if word not in vocabDict:
        vocabDict[word] = 1
    else:
        vocabDict[word] += 1

def track_spam(filename, classType, theClass):
# Tracks Spam and Ham among the documents (this reflects the naming conventions of my specific email set)
    classType.append(theClass)

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def unique_words(a):
    return ' '.join(unique_list(a.split()))

def create_dictionary(fileText, vocabDict, classType, pathHam, pathSpam, testSize, pickling):
# Create dictionary, create documents, track spam and ham, count number of documents
    if pickling == False:
        clear_pickles()

    if not os.path.isfile('counter.obj') or not os.path.isfile('fileText.obj') or not os.path.isfile('classType.obj') or not os.path.isfile('vocabDict.obj'):
        clear_pickles()        
        
        counter = 0
    
        path = pathHam
        for filename in os.listdir(path):
            word_iter_stringio(counter, filename, path, fileText, vocabDict)
            track_spam(filename, classType, 1)
            counter += 1
    
        path = pathSpam
        for filename in os.listdir(path):
            word_iter_stringio(counter, filename, path, fileText, vocabDict)
            track_spam(filename, classType, 0)
            counter += 1

        counter_pick = open('counter.obj', 'w')
        pickle.dump(counter, counter_pick)
        file_pick = open('fileText.obj', 'w')
        pickle.dump(fileText, file_pick)
        class_pick = open('classType.obj', 'w')
        pickle.dump(classType, class_pick)
        vocab_pick = open('vocabDict.obj', 'w')
        pickle.dump(vocabDict, vocab_pick)

    else:
        file_counter = open('counter.obj', 'r')
        counter = pickle.load(file_counter)
        file_file = open('fileText.obj', 'r')
        fileText = pickle.load(file_file)
        file_class = open('classType.obj', 'r')
        classType = pickle.load(file_class)
        file_vocab = open('vocabDict.obj', 'r')
        vocabDict = pickle.load(file_vocab)

    fileText, classType = do_the_shuffle(fileText, classType)
    
    testset = testSize
    if counter < (testset * 2):
        testset = counter / 2
    return (counter, testset, fileText, classType, vocabDict)

def do_the_shuffle(fileText, classType):
    fileText_shuf = []
    classType_shuf = []
    index_shuf = range(len(fileText))
    shuffle(index_shuf)
    
    for i in index_shuf:
        fileText_shuf.append(fileText[i])
        classType_shuf.append(classType[i])
    return (fileText_shuf, classType_shuf)

def clear_pickles():
    if os.path.isfile('counter.obj'):
        os.remove('counter.obj')
    if os.path.isfile('fileText.obj'):
        os.remove('fileText.obj')
    if os.path.isfile('classType.obj'):
        os.remove('classType.obj')
    if os.path.isfile('vocabDict.obj'):
        os.remove('vocabDict.obj')

def sort_dictionary(vocabDict, vocabulary, FEATURES):
# Sorts dictionary by most frequent words
    vocabList = []
    firstSort = sorted(vocabDict.iteritems(), key=lambda item: -item[1])
    vocabFull, counts = zip(*firstSort)
    
    for i in range(FEATURES):
        vocabList.append(vocabFull[i])
    for i in range(FEATURES):
        vocabulary[vocabList[i]] = i

def count_words(counter, fileText, vocabulary, wordCount):
# Determines presence/absence of each word in each document
    for m in range(counter):
        for word in fileText[m].split():
            if word in vocabulary:
                wordCount[m][vocabulary[word]] += 1

def find_theta(counter, testset, FEATURES, classType, wordCount, theta):
# Calculates NBC theta
    for i in range(counter - testset):
        for j in range(FEATURES):
            if classType[i] == 0 and wordCount[i][j] > 0:
                theta[0][j] += 1
            if classType[i] == 1 and wordCount[i][j] > 0:
                theta[1][j] += 1

def almost_equal(x, y, fudge, places=0):
# Equality function used for smoothing purposes - see user-defined variables
    fudgeFactor = 0.65 * fudge
    if not x == 0 and not y == 0:
        y = y / x * fudgeFactor
        x = fudgeFactor
        return round(abs(x-y), places) == 0
    else:
        return x == y

def fine_tune(FEATURES, theta, prior, fudge):
# Eliminating words that don't differentiate between spam and ham
    for j in range(FEATURES):
        if almost_equal((theta[0][j] * prior), (theta[1][j] * (1 - prior)), fudge):
            theta[0][j] = 0
            theta[1][j] = 0

def mutual_information(FEATURES, theta, prior, mutualFeatures, mutualInfo):
# Eliminating words that don't differentiate between spam and ham, using mutual information
    mutualList = []
    for j in range(FEATURES):
        thetaZero = theta[0][j] + 0.000000000000000000000000001
        thetaOne = theta[1][j] + 0.000000000000000000000000001
        thetaJ = (1 - prior) * thetaZero + prior * thetaOne
        termFirstZero = thetaZero * (1 - prior) * math.log(thetaZero / thetaJ)
        termSecondZero = (1 - thetaZero) * prior * math.log((1 - thetaZero) / (1 - thetaJ))
        termFirstOne = thetaOne * prior * math.log(thetaOne / thetaJ)
        termSecondOne = (1 - thetaOne) * (1 - prior) * math.log((1 - thetaOne) / (1 - thetaJ))
        mutualList.append(termFirstZero + termSecondZero + termFirstOne + termSecondOne)
    if mutualFeatures > FEATURES:
        mutualFeatures = FEATURES
    find_n_largest(mutualList, mutualFeatures, mutualInfo, theta)

def find_n_largest(mutualList, n, mutualInfo, theta):
    idxs = [None]*n
    to_watch = range(len(mutualList))
    for i in range(n):
        to_del = 0
        idx = to_watch[to_del]
        max_val = mutualList[idx]
        for jj in range(len(to_watch)):
            j = to_watch[jj]
            val = mutualList[j]
            if val > max_val:
                idx = j
                max_val = val
                to_del = jj
        if theta[0][idx] == 0 and theta[1][idx] == 0:                
            idxs[i] = 0
        else:
            idxs[i] = idx
        del to_watch[to_del]
    mutualInfo.update(idxs)
    mutualInfo.remove(0)

def normalize_theta(counter, testset, FEATURES, theta, prior):
# To normalize theta
    totalZero = 0
    totalOne = 0
    for j in range(FEATURES):
        totalZero += theta[0][j]
        totalOne += theta[1][j]
    for j in range(FEATURES):
        theta[0][j] = theta[0][j] / totalZero
        theta[1][j] = theta[1][j] / totalOne
        
def nbc_train(counter, testset, FEATURES, classType, wordCount, theta, prior, mutualFeatures, mutualInfo, fudge, filtering):
# Training the NBC model
    find_theta(counter, testset, FEATURES, classType, wordCount, theta)
    normalize_theta(counter, testset, FEATURES, theta, prior)
    if filtering:
        fine_tune(FEATURES, theta, prior, fudge)
        mutual_information(FEATURES, theta, prior, mutualFeatures, mutualInfo)

def logsumexp(i, FEATURES, prob, theta, mutualInfo, sum_exp_Zero, sum_exp_One, wordCount, filtering):
    for j in range(FEATURES):
        if (j in mutualInfo and filtering) or not filtering:
            if wordCount[i][j] == 0:
                prob[0][i] += math.log(1 - theta[0][j])
                sum_exp_Zero += math.exp(math.log(1 - theta[0][j]))
                prob[1][i] += math.log(1 - theta[1][j])
                sum_exp_One += math.exp(math.log(1 - theta[1][j]))
            if (wordCount[i][j] > 0):
                prob[0][i] += (math.log(theta[0][j] + 0.00000000000000001)) * wordCount[i][j]
                sum_exp_Zero += math.exp(math.log(theta[0][j] + 0.00000000000000001)) * wordCount[i][j]
                prob[1][i] += (math.log(theta[1][j] + 0.00000000000000001)) * wordCount[i][j]
                sum_exp_One += math.exp(math.log(theta[1][j] + 0.00000000000000001)) * wordCount[i][j]

    prob[0][i] = prob[0][i] - math.log(sum_exp_Zero)
    prob[1][i] = prob[1][i] - math.log(sum_exp_One)

def doc_result(i, prob, totalProb, classType, finalCount, zeroCount, oneCount, hamCount):
    if prob[1][i] > prob[0][i]:  # Probability of ham vs spam
        totalProb[i] = 1
    if classType[i] == totalProb[i]:
        finalCount += 1
        if classType[i] == 0:
            zeroCount += 1
        else:
            oneCount += 1
    if classType[i] == 1:
        hamCount += 1
    print "Document: " + str(i)
    print "Class: " + str(classType[i])
    if round(abs((prob[0][i] + (prob[1][i]))), 7) != 0:
        print "Prob Class 1 (Ham): " + str("{0:.2%}".format(prob[1][i] / (prob[0][i] + (prob[1][i]))))
        print "Prob Class 0 (Spam): " + str("{0:.2%}".format(prob[0][i] / (prob[0][i] + (prob[1][i]))))
    else:
        # working with floating-point, we need to avoid a division-by-zero
        print "Predicted Class: " + str(totalProb[i])
    if classType[i] == totalProb[i]:
        print "Correct!"
    else:
        print "Incorrect"
    print
    
    return (finalCount, zeroCount, oneCount, hamCount)

def calculate_prob(counter, testset, FEATURES, wordCount, classType, theta, prob, prior, totalProb, mutualInfo, filtering):
# Tests each document in the test set, prints probability of spam vs ham
    finalCount = 0
    zeroCount = 0
    oneCount = 0
    hamCount = 0
    priorZero = math.exp(1 - prior)
    priorOne = math.exp(prior)

    for i in range(counter-testset, counter):
        prob[0][i] = priorZero
        prob[1][i] = priorOne
        sum_exp_Zero = 1 - prior
        sum_exp_One = prior
        
        logsumexp(i, FEATURES, prob, theta, mutualInfo, sum_exp_Zero, sum_exp_One, wordCount, filtering)

        finalCount, zeroCount, oneCount, hamCount = doc_result(i, prob, totalProb, classType, finalCount, zeroCount, oneCount, hamCount)

    return (finalCount, zeroCount, oneCount, hamCount)


def nbc_test(counter, testset, FEATURES, wordCount, classType, theta, prior, mutualInfo, filtering):
# This is where the testing happens!    
    prob = [[1.0]*counter for i in range(2)]
    totalProb = [0]*counter
    
    finalCount, zeroCount, oneCount, hamCount = calculate_prob(counter, testset, FEATURES, wordCount, classType, theta, prob, prior, totalProb, mutualInfo, filtering)
    spamCount = testset - hamCount

    print "Test Accuracy = " + "{:.2%}".format(float(finalCount) / testset)
    print "Fraction Correct: " + str(finalCount) + " / " + str(testset)
    print "Correct Ham Samples: " + str(oneCount) + " / " + str(hamCount)
    print "Correct Spam Samples: " + str(zeroCount) + " / " + str(spamCount)
    print
    print "Thanks for running your friendly neighborhood NBC Spam Detector!"
    print


def main(args):
    try:
        # Defining the structures
        vocabDict, vocabulary = {}, {}
        classType, fileText = [], []
        mutualInfo = set()
        
        ''' To be altered by the user as needed '''
        ''' ----------------------------------- '''
        FEATURES = 2500 # Features (words) selected by number of appearances in all documents
        
        filtering = False # If true, will use mutualFeatures and fudge below for filtering; if false, no filtering performed
        mutualFeatures = 5000 # Set at number of features to keep, as ranked by mutual information
        fudge = 1 # Set at a minimum of 1; at 1, this smooths out as many of the
        # "less important features" as possible; no smoothing applied at very high numbers

        testSize = 300
        
        pathHam = "./Ham/" # Path for Ham documents
        pathSpam = "./Spam/" # Path for Spam documents

        pickling = True # Pickles the create_dictionary step up to randomization step, to shorten program running time
        ''' ----------------------------------- '''

        counter, testset, fileText, classType, vocabDict = create_dictionary(fileText, vocabDict, classType, pathHam, pathSpam, testSize, pickling)
        sort_dictionary(vocabDict, vocabulary, FEATURES)

        wordCount = [[0]*FEATURES for i in range(counter)]
        prior = float(sum(classType[i] for i in range(counter))) / (counter)
        theta = [[0.0]*FEATURES for i in range(2)]

        count_words(counter, fileText, vocabulary, wordCount)
        nbc_train(counter, testset, FEATURES, classType, wordCount, theta, prior, mutualFeatures, mutualInfo, fudge, filtering)
        nbc_test(counter, testset, FEATURES, wordCount, classType, theta, prior, mutualInfo, filtering)

    except Exception, err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        return 1
    else:
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
