import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec, TaggedDocument
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
    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    features = set()

    def add_words_to_features(data):
      word_dict = {}
      for line in data:
        word_count_by_line = Counter(line)
        for k,v in word_count_by_line.iteritems():
          if k not in stopwords:
            word_dict[k] = word_dict.get(k,0) + 1
      for k in word_dict.keys():
        if word_dict[k] >= 0.01*len(data):
          features.add(k)
      return word_dict

    def check_for_twice(item):
      if (item in pos_dict and pos_dict[item] >= 2*neg_dict.get(item,0)):
        return True
      elif (item in neg_dict and neg_dict[item] >= 2*pos_dict.get(item,0)):
        return True

    pos_dict = add_words_to_features(train_pos)
    neg_dict = add_words_to_features(train_neg)

    features = filter(check_for_twice,features)

    #print len(features)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.

    def create_features_vector(document):
      features_vector=[]
      for line in document:
        features_vector.append([1 if word in line else 0 for word in features])
      return features_vector

    train_pos_vec = create_features_vector(train_pos)
    train_neg_vec = create_features_vector(train_neg)
    test_pos_vec  = create_features_vector(test_pos)
    test_neg_vec  = create_features_vector(test_neg)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    def create_labeled_sentences(data,label):
      temp = []
      for i,row in enumerate(data):
        temp.append(TaggedDocument(words=row, tags =[label+str(i)]))
      return temp
    def transform(model, label,data):
      out = []
      for i, row in enumerate(data):
          out.append(model.docvecs[label+str(i)])
      return out

    labeled_train_pos=create_labeled_sentences(train_pos,"TRAIN_POS_")
    labeled_train_neg=create_labeled_sentences(train_neg,"TRAIN_NEG_")
    labeled_test_pos=create_labeled_sentences(test_pos,"TEST_POS_")
    labeled_test_neg=create_labeled_sentences(test_neg,"TEST_NEG_")

    # Initialize model
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
    train_pos_vec = transform(model,"TRAIN_POS_",train_pos)
    train_neg_vec = transform(model,"TRAIN_NEG_",train_neg)
    test_pos_vec = transform(model, "TEST_POS_", test_pos)
    test_neg_vec = transform(model,"TEST_NEG_",test_neg)

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
    train_data = train_pos_vec+train_neg_vec

    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(train_data, Y)

    lr_model = LogisticRegression()
    lr_model.fit(train_data, Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    train_data = train_pos_vec + train_neg_vec

    nb_model = GaussianNB()
    nb_model.fit(train_data, Y)

    lr_model = LogisticRegression()
    lr_model.fit(train_data,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    tp, fn, fp, tn = 0, 0, 0, 0
    predict_pos = model.predict(test_pos_vec)
    predict_neg = model.predict(test_neg_vec)

    for each in predict_pos:
      if each == "pos":
        tp +=1
      else:
        fn +=1

    for each in predict_neg:
      if each == "neg":
        tn +=1
      else:
        fp +=1

    accuracy = (tp+tn)/float(tp+tn+fn+fp)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
