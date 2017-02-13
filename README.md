# Mixed-membership Recommender
Mixed-membership Recommender based on this paper:<br/> 
<a href="https://arxiv.org/abs/1604.01170">Accurate and scalable recommendation using mixed-membership stochastic block models</a>

# Alternative to Matrix Factorization
This approach is an alternative to the matrix factorization approach currently implemented in Spark's
recommender library.  The mixed-membership model is based on multiple groups of users and items.  Users
 and items can each belong to mixtures of different groups.  A rating of an item by a user
 is determined by the membership mixtures of both the item and the user.  In short, it is a more 
 explicit approach than matrix factorization.  A more complete and accurate description of this
 approach can be found in the paper referenced above.
 
# Built for Spark
This implementation is built specifically for Apache Spark.  It leverages Spark's pipeline API, so that the 
library can seamlessly be integrated with existing Spark-based machine learning processes.  This means that 
it uses and Estimator and Transformer for learning and prediction, respectively.

# Usage
```
    def mapRatingRow(line):
        charDelim = '\t'
        user, movie, rating, timestamp = line.split(charDelim, 3)
        return Row(user=int(user), movie=int(movie), rating=float(rating))

    # load data file
    ratingsRDD = sc.textFile("movie.100k.data")
    rddTrain, rddTest = ratingsRDD.randomSplit([.7, .3], 17)
    rddTrain = rddTrain.map(mapRatingRow)
    rddTest = rddTest.map(mapRatingRow)
    ratings = spark.createDataFrame(rddTrain)
    test_ratings = spark.createDataFrame(rddTest)

    # fit model
    rec = MixMemRecommender(itemCol='movie', numUserGroups=10, numItemGroups=10)
    model = rec.fit(data)
    
    # predict with model
    predictions = model.transform(test_ratings)
    
```

