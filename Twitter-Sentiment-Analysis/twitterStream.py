from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():

    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist(sc, "positive.txt")
    nwords = load_wordlist(sc, "negative.txt")

    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)

def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    print counts
    plt.xlabel("Time Step")
    plt.ylabel("Word Count")
    plt.axis([0,12,0,300])

    positive_count=[]
    negative_count=[]

    for count in counts:
        if count:
            positive_count.append(count[0][1])
            negative_count.append(count[1][1])

    plt.plot(range(len(positive_count)),positive_count,label='positive',marker='o')
    plt.plot(range(len(negative_count)),negative_count,label='negative',marker='o')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(1, len(counts), 1))
    plt.show()



def load_wordlist(sc, filename):
    """
    This function should return a list or set of words from the given filename.
    """
    return sc.textFile(filename).collect()

def updateFunction(newValues, runningCount):
    if runningCount is None:
       runningCount = 0
    return sum(newValues, runningCount)

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    allWords = tweets.flatMap(lambda line: line.split(" "))
    positive_words=allWords.filter(lambda x: x.lower() in pwords)
    negative_words=allWords.filter(lambda x: x.lower() in nwords)

    positive_pairs=positive_words.map(lambda x: ('positive',1))
    negative_pairs=negative_words.map(lambda x: ('negative',1))

    positiveCounts=positive_pairs.reduceByKey(lambda x, y: x + y)
    negativeCounts=negative_pairs.reduceByKey(lambda x, y: x + y)

    totalCounts=positiveCounts.union(negativeCounts)
    runningCounts = totalCounts.updateStateByKey(updateFunction)

    runningCounts.pprint()

    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []

    totalCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))


    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
