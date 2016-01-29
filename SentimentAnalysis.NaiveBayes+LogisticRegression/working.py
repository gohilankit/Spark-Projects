from __future__ import division
import sys
import collections
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec, TaggedDocument
import pdb
from collections import Counter
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    pos_term_dict, neg_term_dict, pos_file_dict, neg_file_dict = {}, {}, {}, {}
    candidate_features = set()
    features = []
    def mycount(data):
        temp_dict = {}
        file_dict = {}
        for row in data:
            row_counts = Counter(row)
            for word, counts in row_counts.iteritems():
                if word in stopwords:
                    continue
                temp_dict[word] = temp_dict.get(word,0) +1
                file_dict[word] = file_dict.get(word,0) +1

        for word, count in file_dict.iteritems():
            if count >= len(data)*0.01:
                candidate_features.add(word)

        return (temp_dict, file_dict)

    def create_features_vec(features,txt):
        new_vect = []
        for row in txt:
            new_vect.append([1 if each in row else 0 for each in features])
        return new_vect


    pos_term_dict, pos_file_dict = mycount(train_pos)
    neg_term_dict, neg_file_dict = mycount(train_neg)
    for each in candidate_features:
        if each in pos_term_dict.keys():
            if each in neg_file_dict.keys():
                if pos_file_dict[each] >= 2* neg_file_dict[each]:
                    features.append(each)
            else: # if this feature does not exist in negative text
                features.append(each)
        if each in neg_term_dict.keys():
            if each in pos_term_dict.keys():
                if neg_file_dict[each] >= 2* pos_file_dict[each]:
                    features.append(each)
            else: # if this feature does not exist in positive text
                features.append(each)

    train_pos_vec = create_features_vec(features,train_pos)
    train_neg_vec = create_features_vec(features,train_neg)
    test_pos_vec = create_features_vec(features,test_pos)
    test_neg_vec = create_features_vec(features,test_neg)
    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    # Initialize model
    def prepare(data, label):
      temp = []
      for i,row in enumerate(data):
        temp.append(TaggedDocument(words=row, tags =[label+str(i)]))
      return temp
    def transform(model, label,data):
        out = []
        for i, row in enumerate(data):
            out.append(model.docvecs[label+str(i)])
        return out


    labeled_train_pos = prepare(train_pos,"TRAIN_POS_")
    labeled_train_neg = prepare(train_neg,"TRAIN_NEG_")
    labeled_test_pos = prepare(test_pos, "TEST_POS_")
    labeled_test_neg = prepare(test_neg, "TEST_NEG_")
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = transform(model,"TRAIN_POS_",train_pos)
    train_neg_vec = transform(model,"TRAIN_NEG_",train_neg)
    test_pos_vec = transform(model, "TEST_POS_", test_pos)
    test_neg_vec = transform(model,"TEST_NEG_",test_neg)
    # pdb.set_trace()
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    train = train_pos_vec+train_neg_vec
    lr_model = LogisticRegression()
    lr_model.fit(train, Y)
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(train, Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    train = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model.fit(train, Y)
    lr_model = LogisticRegression()
    lr_model.fit(train,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE

    tp, fn, fp, tn = 0, 0, 0, 0
    predict_pos = model.predict(test_pos_vec)
    predict_neg = model.predict(test_neg_vec)

    for each in predict_pos:
      if each == "pos":
        tp +=1
      else:
        fp +=1

    for each in predict_neg:
      if each == "neg":
        tn +=1
      else:
        fn +=1

    accuracy = (tp+tn)/(tp+tn+fn+fp)


    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
