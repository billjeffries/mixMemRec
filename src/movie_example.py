from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import Row
from pyspark.mllib.evaluation import RankingMetrics


def createMovieRatingPair(pred):
    return pred.user, (pred.movie, pred.rating)


def createMoviePredictionPair(pred):
    return pred.user, (pred.movie, pred.prediction)


def sortTuples(movies):
    uid = movies[0]
    mTuples = movies[1]
    sortedMovies = sorted(mTuples, key=lambda x: x[1])
    rankedMovies = []
    for tup in sortedMovies:
        if (tup[1] >= 3.0):
            rankedMovies.append(tup[0])
    return uid, rankedMovies


def mapRatingRow(line):
    charDelim = '\t'
    user, movie, rating, timestamp = line.split(charDelim, 3)
    return Row(user=int(user), movie=int(movie), rating=float(rating))


def calculateMetrics(predictionDF):
    metricOutput = []
    # Calculate regression metrics
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictionDF)
    metricOutput.append({"metric": "RMSE", "value": str(rmse)})
    evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    mse = evaluator.evaluate(predictionDF)
    metricOutput.append({"metric": "MAE", "value": str(mse)})

    # Calculate accuracy metrics
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="rating")
    acc = evaluator.evaluate(predictionDF, {evaluator.metricName: "accuracy"})
    metricOutput.append({"metric": "Accuracy", "value": str(acc)})

    # Calculate ranking metrics
    testRDD = predictionDF.rdd.map(createMovieRatingPair) # user, (movie, rating)
    testRDD = testRDD.groupByKey() # user, [(movie1, rating1), (movie2, rating2), ..., (movieN, ratingN)]
    testRDD = testRDD.map(sortTuples) # user, [movie1, movie2, ... , movieN)

    predictionsRDD = predictionDF.rdd.map(createMoviePredictionPair) # user, (movie, prediction)
    predictionsRDD = predictionsRDD.groupByKey() # user [(movie1, prediction1), (movie2, prediction2)...]
    predictionsRDD = predictionsRDD.map(sortTuples) # user, [movie1, movie2, ... , movieN]
    predictionsAndRatings = predictionsRDD.join(testRDD).values()

    metrics = RankingMetrics(predictionsAndRatings)
    map = metrics.meanAveragePrecision
    metricOutput.append({"metric": "MAP", "value": str(map)})

    # Output metrics
    metricRDD = sc.parallelize(metricOutput)
    metricDF = metricRDD.toDF()
    metricDF.coalesce(1).write.format('json').save("rec-metrics.json")


def train_model(data):
    # fit model
    rec = MixMemRecommender(itemCol='movie')
    recModel = rec.fit(data)
    return recModel


def predict_with_model(recModel, data):

    predictions = recModel.transform(test_ratings)
    # calculate metrics
    calculateMetrics(predictions)


if __name__ == "__main__":
    conf = SparkConf()
    conf.set('spark.executor.memory', '8g')
    conf.set('spark.app.name', 'mix-mem-rec')
    conf.set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')

    sc = SparkContext(conf=conf)
    spark = SparkSession \
        .builder \
    .getOrCreate()

    sc.addFile("mixmemrec.py")
    from mixmemrec import MixMemRecommender, MixMemModel

    # load data file
    ratingsRDD = sc.textFile("movie.100k.data")
    rddTrain, rddTest = ratingsRDD.randomSplit([.7, .3], 17)
    rddTrain = rddTrain.map(mapRatingRow)
    rddTest = rddTest.map(mapRatingRow)
    ratings = spark.createDataFrame(rddTrain)
    test_ratings = spark.createDataFrame(rddTest)

    # train model
    model = train_model(ratings)

    # save model
    model.save("mixmemrec")

    # load model
    model = MixMemModel(itemCol='movie')
    model.load(spark, "mixmemrec")

    # make predictions
    predict_with_model(model, test_ratings)




