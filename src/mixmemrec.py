from pyspark import StorageLevel
from pyspark.ml import Estimator, Transformer
from pyspark.sql import Row
import numpy as np
import math
import random
from datetime import datetime
import operator
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, ArrayType, DoubleType


class MixMemModel(Transformer):
    def __init__(self, itemCol="item",
                        userCol="user",
                        ratingCol="rating",
                        ratingScale=[1, 2, 3, 4, 5],
                        userMemberships=None,
                        itemMemberships=None,
                        probabilities=None,
                        avgUserMembership=None):
        self.itemCol = itemCol
        self.userCol = userCol
        self.ratingCol = ratingCol
        self.ratings = ratingScale
        self.userMemberships = userMemberships
        self.itemMemberships = itemMemberships
        self.probabilities = probabilities
        self.avgUserMembership = avgUserMembership

    def getPaths(self, path):
        userMembershipPath = "{0}/userMemberships".format(path)
        itemMembershipPath = "{0}/itemMemberships".format(path)
        probabilitiesPath = "{0}/probs".format(path)
        avgUserPath = "{0}/avgUser".format(path)
        ratingScalePath = "{0}/ratingScale".format(path)
        return userMembershipPath, itemMembershipPath, probabilitiesPath, avgUserPath, ratingScalePath

    def save(self, path):
        userPath, itemPath, probPath, avgUPath, ratingScalePath = self.getPaths(path)

        self.userMemberships.write.parquet(userPath)
        self.itemMemberships.write.parquet(itemPath)

        sc = self.userMemberships.rdd.context

        probs = sc.parallelize(self.probabilities)
        probs.saveAsPickleFile(probPath)

        avgUser = sc.parallelize(self.avgUserMembership)
        avgUser.saveAsPickleFile(avgUPath)

        ratingScale = sc.parallelize(self.ratings)
        ratingScale.saveAsPickleFile(ratingScalePath)

    def load(self, spark, path):
        userPath, itemPath, probPath, avgUPath, ratingScalePath = self.getPaths(path)

        self.userMemberships = spark.read.parquet(userPath)
        self.itemMemberships = spark.read.parquet(itemPath)

        sc = self.userMemberships.rdd.context

        probs = sc.pickleFile(probPath)
        self.probabilities = probs.collect()

        avgUser = sc.pickleFile(avgUPath)
        self.avgUserMembership = avgUser.collect()

        ratingScale = sc.pickleFile(ratingScalePath)
        self.ratings = ratingScale.collect()

    def _transform(self, dataset):
        def predict(user, item, uMem, iMem, rating, probs, avgUser, ratings):
            p_shape = probs[0][1].shape
            numUserGroups = p_shape[0]
            numItemGroups = p_shape[1]
            predSums = []
            if uMem is None:
                uMem = avgUser
            for r in ratings:
                rValues = probs[r-1][1]
                rSum = 0
                for k in range(numUserGroups):
                    for l in range(numItemGroups):
                        p = rValues[k][l]
                        uk = uMem[k]
                        nl = iMem[l]
                        rSum += uk * nl * p
                predSums.append(rSum)
            index, v = max(enumerate(predSums), key=operator.itemgetter(1))
            rHat = ratings[index]
            return float(rHat)

        def predict_probabilities(probs, avgUser, ratings):
            return udf(lambda u, i, um, im, r: predict(u, i, um, im, r, probs, avgUser, ratings), DoubleType())

        # build prediction data frame
        df = dataset.join(self.userMemberships, dataset[self.userCol] == self.userMemberships.user, 'left_outer')\
                    .drop(self.userMemberships.user)
        df = df.join(self.itemMemberships, df[self.itemCol] == self.itemMemberships.item)\
            .drop(self.itemMemberships.item)

        # predict
        predDF = df.withColumn("prediction",
                               predict_probabilities(self.probabilities, self.avgUserMembership, self.ratings)
                                                    (df[self.userCol],
                                                        df[self.itemCol],
                                                        df['uMem'],
                                                        df['iMem'],
                                                        df[self.ratingCol]))

        return predDF


class MixMemRecommender(Estimator):
    def __init__(self, itemCol="item",
                        userCol="user",
                        ratingCol="rating",
                        ratingScale=[1, 2, 3, 4, 5],
                        maxIter=300,
                        numUserGroups=10,
                        numItemGroups=10):
        self.itemCol = itemCol
        self.userCol = userCol
        self.ratingCol = ratingCol
        self.ratings = ratingScale
        self.maxIter = maxIter
        self.numUserGroups = numUserGroups
        self.numItemGroups = numItemGroups

    def mapRatingRow(self, record):
        user = record[0]
        item = record[1]
        rating = record[2]
        return (user, item), rating

    def mapUserItems(self, record):
        key = record[0]
        user = key[0]
        item = key[1]
        return (user, item)

    def mapItemUsers(self, record):
        key = record[0]
        user = key[0]
        item = key[1]
        return (item, user)

    def emitUser(self, userRating):
        key = userRating[0]
        user = key[0]
        return user, 1

    def createUserMemberships(self, users):
        user = users[0]
        user_memberships = []
        min = 0
        max = 1
        random.seed(datetime.now())
        for k in range(self.numUserGroups):
            rand_prob = random.uniform(min, max)
            user_memberships.append(rand_prob)
            max = max - rand_prob

        return user, user_memberships

    def emitItem(self, userRating):
        key = userRating[0]
        item = key[1]
        return item, 1

    def createItemMemberships(self, items):
        item = items[0]
        item_memberships = []
        min = 0
        max = 1
        random.seed(datetime.now())
        for l in range(self.numItemGroups):
            rand_prob = random.uniform(min, max)
            item_memberships.append(rand_prob)
            max = max - rand_prob

        return item, item_memberships

    def initializeProbabilities(self):
        probs = []
        for r in self.ratings:
            values = np.zeros((self.numUserGroups, self.numItemGroups))
            probs.append((r, values))

        for k in range(self.numItemGroups):
            for l in range(self.numItemGroups):
                min = 0
                max = 1
                random.seed(datetime.now())
                for r in self.ratings:
                    rand_prob = random.uniform(min, max)
                    probs[r-1][1][k][l] = rand_prob
                    max = max - rand_prob

        return probs

    def updateWeights(self, partition):
        probs = probabilities.value

        for record in partition:
            key = record[0]
            value = record[1]
            user = key[0]
            item = key[1]
            rating = int(value)

            user_memberships = userMemberships.value[user]
            item_memberships = itemMemberships.value[item]

            norm = 0
            for k_prime in range(self.numUserGroups):
                uk_prime = user_memberships[k_prime]
                for l_prime in range(self.numItemGroups):
                    il_prime = item_memberships[l_prime]
                    pkl_prime = probs[rating-1][1][k_prime][l_prime]
                    norm += uk_prime * il_prime * pkl_prime
            new_weights = []
            sum = 0.0
            for k in range(self.numUserGroups):
                uk = user_memberships[k]
                for l in range(self.numItemGroups):
                    il = item_memberships[l]
                    pkl = probs[rating-1][1][k][l]
                    liklihood = uk * il * pkl
                    sum += liklihood
                    new_weight = liklihood / norm
                    new_weights.append((k+1, l+1, new_weight))
            yield (user, item), (rating, new_weights, sum)

    def userWeights(self, record):
        key = record[0]
        value = record[1]
        user = key[0]
        weights = value[1]
        sums = {}
        for klw in weights:
            k = klw[0]
            w = klw[2]
            if k not in sums:
                sums[k] = 0
            sums[k] += w
        sum_array = []
        for i in range(self.numUserGroups):
            k = i+1
            sum_array.append(sums[k])

        return (user, sum_array)

    def sumUserWeights(self, v1, v2):
        s1 = np.array(v1)
        s2 = np.array(v2)
        wsum = np.add(s1, s2)
        return wsum

    def itemWeights(self, record):
        key = record[0]
        value = record[1]
        item = key[1]
        weights = value[1]
        sums = {}
        for klw in weights:
            l = klw[1]
            w = klw[2]
            if l not in sums:
                sums[l] = 0
            sums[l] += w
        sum_array = []
        for i in range(self.numItemGroups):
            l = i+1
            sum_array.append(sums[l])

        return (item, sum_array)

    def sumItemWeights(self, v1, v2):
        s1 = np.array(v1)
        s2 = np.array(v2)
        wsum = np.add(s1, s2)
        return wsum

    def averageUserWeights(self, sum_weights, num_weights):
        avg_weights = np.divide(sum_weights, num_weights)
        return avg_weights

    def updateUserMemberships(self, record):
        user = record[0]
        items = userItemMap.value[user]
        num_ratings = len(items)
        user_weights = record[1]
        new_weights = []
        for k in range(self.numUserGroups):
            sum_k = user_weights[k]
            new_weights.append(sum_k / float(num_ratings))
        return (user, new_weights)

    def updateItemMemberships(self, record):
        item = record[0]
        users = itemUserMap.value[item]
        num_ratings = len(users)
        item_weights = record[1]
        new_weights = []
        for l in range(self.numItemGroups):
            sum_l = item_weights[l]
            new_weights.append(sum_l / float(num_ratings))
        return (item, new_weights)

    def sumWeights(self, v1, v2):
        s1 = np.array(v1)
        s2 = np.array(v2)
        wsum = np.add(s1, s2)
        return wsum

    def normalizeWeights(self, weights):
        n_weights = []
        for w in weights:
            key = w[0]
            r_weights = w[1]
            new_weights = []
            sum = 0
            for rw in r_weights:
                sum += rw
            for rating in self.ratings:
                new_value = r_weights[rating-1] / sum
                new_weights.append((rating, new_value))
            n_weights.append((key, new_weights))
        return n_weights

    def updateProbabilites(self, r_weights):
        probs = []
        ratings = {}
        for rw in r_weights:
            k = rw[0][0]
            l = rw[0][1]
            r_values = rw[1]
            for rv in r_values:
                rating = rv[0]
                value = rv[1]
                if rating not in ratings:
                    ratings[rating] = []
                ratings[rating].append((k, l, value))

        for r in self.ratings:
            values = np.zeros((self.numUserGroups, self.numItemGroups))
            if r in ratings:
                kl_values = ratings[r]
                for klv in kl_values:
                    k2 = klv[0]
                    l2 = klv[1]
                    v = klv[2]
                    values[k2-1][l2-1] = v
            probs.append((r, values))

        return probs

    def multiplyLikelihoods(self, a, b):
        if b < 0:
            return b
        else:
            value = a + math.log(b)
            return value

    def extractGroupKey(self, record):
        value = record[1]
        rating = value[0]
        weights = list(value[1])
        groups = []
        for w in weights:
            k = w[0]
            l = w[1]
            v = w[2]
            values = [0, 0, 0, 0, 0]
            values[rating-1] = v
            groups.append(((k, l), values))
        return groups

    def userMemRow(self, record):
        user = record[0]
        weights = record[1]
        weights = [float(x) for x in weights]
        return Row(user=int(user), uMem=weights)

    def itemMemRow(self, record):
        item = record[0]
        weights = record[1]
        weights = [float(x) for x in weights]
        return Row(item=int(item), iMem=weights)

    def _fit(self, training):
        # convert ratings df to rdd
        ratingsRDD = training.select(self.userCol, self.itemCol, self.ratingCol).rdd
        ratingsRDD = ratingsRDD.map(self.mapRatingRow) # (u, i), r
        ratingsRDD.cache() # rdd will be used multiple times below

        # get SparkContext
        sc = ratingsRDD.context

        # create membership rdds
        userItemRDD = ratingsRDD.map(self.mapUserItems).groupByKey().cache() # u, [items]
        global userItemMap
        userItemMap = sc.broadcast(userItemRDD.collectAsMap())
        itemUserRDD = ratingsRDD.map(self.mapItemUsers).groupByKey().cache() # i, [users]
        global itemUserMap
        itemUserMap = sc.broadcast(itemUserRDD.collectAsMap())

        # initialize
        weightsRDD = None
        previousWeightsRDD = None
        userRDD = ratingsRDD.map(self.emitUser).groupByKey().map(self.createUserMemberships).persist(StorageLevel.MEMORY_AND_DISK) # u, [(k,v)...(k,v)]
        itemRDD = ratingsRDD.map(self.emitItem).groupByKey().map(self.createItemMemberships).persist(StorageLevel.MEMORY_AND_DISK) # i, [(l,v)...(l,v)]
        global probabilities
        probabilities = sc.broadcast(self.initializeProbabilities())

        lastLikelihood = 0
        for i in range(self.maxIter):
            print("Iteration {0}".format(str(i+1)))

            # update weights
            if weightsRDD is not None:
                previousWeightsRDD = weightsRDD
            userMemMap = userRDD.collectAsMap()
            global userMemberships
            userMemberships = sc.broadcast(userMemMap)
            itemMemMap = itemRDD.collectAsMap()
            global itemMemberships
            itemMemberships = sc.broadcast(itemMemMap)
            weightsRDD = ratingsRDD.mapPartitions(self.updateWeights).persist(StorageLevel.MEMORY_AND_DISK) # (u, i), (r, [weights], likelihood)
            if previousWeightsRDD is not None:
                previousWeightsRDD.unpersist()

            # update user membership
            previousUserRDD = userRDD
            userRDD = weightsRDD.map(self.userWeights)
            userRDD = userRDD.reduceByKey(self.sumUserWeights) # u, [weights]
            userRDD = userRDD.map(self.updateUserMemberships).persist(StorageLevel.MEMORY_AND_DISK) # u, [memberships]
            previousUserRDD.unpersist()

            # update item membership
            previousItemRDD = itemRDD
            itemRDD = weightsRDD.map(self.itemWeights)
            itemRDD = itemRDD.reduceByKey(self.sumItemWeights)
            itemRDD = itemRDD.map(self.updateItemMemberships).persist(StorageLevel.MEMORY_AND_DISK) # i, [memberships]
            previousItemRDD.unpersist()

            # update probabilities
            ratingRDD = weightsRDD.flatMap(self.extractGroupKey) # (k,l), [0, 0, 0, v, 0]
            ratingRDD = ratingRDD.reduceByKey(self.sumWeights).persist(StorageLevel.MEMORY_AND_DISK) # (k,l), [sum1, sum2, ..., sum5]
            ratingWeights = ratingRDD.collect()
            ratingWeights = self.normalizeWeights(ratingWeights)
            probabilities = sc.broadcast(self.updateProbabilites(ratingWeights))
            ratingRDD.unpersist()

            # calculate likelihood
            ratingLikes = weightsRDD.map(lambda x: x[1][2]).persist(StorageLevel.MEMORY_AND_DISK) # likelihood
            likelihood = ratingLikes.fold(0, self.multiplyLikelihoods)
            print("Likelihood = {0}".format(str(likelihood)))
            ratingLikes.unpersist()

            # check convergence
            converged = False
            if lastLikelihood < 0:
                likeDiff = abs(lastLikelihood - likelihood)
                print("change = {0}".format(str(likeDiff)))
                if float(likeDiff) / float(abs(likelihood)) < .001:
                    converged = True
            if converged:
                break
            lastLikelihood = likelihood

        # Calculate average user membership for cold-start
        sum_user_mem = userRDD.map(lambda x: x[1]).reduce(self.sumUserWeights)
        num_user_mem = userRDD.count()
        avg_user_mem = self.averageUserWeights(sum_user_mem, num_user_mem)

        # Create dataframes
        userDF = userRDD.map(self.userMemRow).toDF()
        itemDF = itemRDD.map(self.itemMemRow).toDF()

        # Return trained model
        model = MixMemModel(itemCol=self.itemCol,
                            userCol=self.userCol,
                            ratingCol=self.ratingCol,
                            userMemberships=userDF,
                            itemMemberships=itemDF,
                            probabilities=probabilities.value,
                            avgUserMembership=avg_user_mem)
        return model
