import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object ModelHelper {

    def getJaccardSimilarity(commonFriends: Int, friendsCount1:Int, friendsCount2:Int) : Double = {
        val union = friendsCount1 + friendsCount2 - commonFriends

        if (union == 0) {
            0.0
        } else {
            commonFriends.toDouble / union.toDouble
        }
    }

    def getCosineSimilarity(commonFriends: Int, friendsCount1:Int, friendsCount2:Int) : Double = {
        if (friendsCount1 == 0 && friendsCount2 == 0) {
            0.0
        } else {
            commonFriends.toDouble / math.sqrt(friendsCount1 * friendsCount2)
        }
    }

    def createFeatures(pair: PairWithScore,
                       ageSexBC: Broadcast[scala.collection.Map[Int, AgeSexCity]],
                       cityPairCountBC: Broadcast[scala.collection.Map[(Int, Int), Int]],
                       coreUserFriendCountBC: Broadcast[scala.collection.Map[Int, Int]]) : Vector = {

        val ageSexCity1 = ageSexBC.value.getOrElse(pair.person1, AgeSexCity(0, 0, 0))
        val ageSexCity2 = ageSexBC.value.getOrElse(pair.person2, AgeSexCity(0, 0, 0))
        val isSameSex = ageSexCity1.sex == ageSexCity2.sex
        val city1 = ageSexCity1.city
        val city2 = ageSexCity2.city
        val cityFactor = if (city1 == city2) {
            1.0
        } else {
            val cityPair = if (city1 < city2) (city1, city2) else (city2, city1)
            val cityCount = cityPairCountBC.value.getOrElse(cityPair, 0)
            if (cityCount > 50000) {
                0.5
            } else {
                0.0
            }
        }


        val diffAge = abs(ageSexCity1.age - ageSexCity2.age).toDouble
        val signAge = if (diffAge < 300.0) 0.0 else if (ageSexCity2.age > ageSexCity1.age) 1.0 else -1.0
        val positiveDiffAge = if (ageSexCity2.age > ageSexCity1.age) diffAge else 0.0
        val negativeDiffAge = if (ageSexCity2.age < ageSexCity1.age) diffAge else 0.0

        val meanAge = 25000.0 - (ageSexCity1.age + ageSexCity2.age) * 0.5
        val friendsCount1 = coreUserFriendCountBC.value.getOrElse(pair.person1, 0)
        val friendsCount2 = coreUserFriendCountBC.value.getOrElse(pair.person2, 0)
        val jaccard = getJaccardSimilarity(pair.commonFriendsCount, friendsCount1, friendsCount2)
        val cosine = getCosineSimilarity(pair.commonFriendsCount, friendsCount1, friendsCount2)

        Vectors.dense(
            Math.log(1.0 + pair.aaScore),
            Math.log(1.0 + pair.fedorScore),
            Math.log(positiveDiffAge + 1.0),
            Math.log(negativeDiffAge + 1.0),
            signAge,
            Math.log(meanAge),
            jaccard,
            cosine,
            Math.log(1.0 + pair.interactionScore),
            if (isSameSex) 1.0 else 0.0,
            cityFactor,
            pair.isStrongRelation,
            pair.isWeakRelation,
            pair.isColleague,
            pair.isSchoolmate,
            pair.isArmyFellow,
            pair.isOther,
            if (pair.maskAnd > 1) 1.0 else 0.0
        )
    }

    /**
      * Generates RDD of pairs with features and labels.
      * Pairs are ordered (UID1 < UID2)
      *
      * (UID1, UID2) -> (Vector_of_features, Label)
      */
    def prepareData(
                       pairScores: RDD[PairWithScore],
                       positives: RDD[((Int, Int), Double)],
                       ageSexBC: Broadcast[scala.collection.Map[Int, AgeSexCity]],
                       cityPairCountBC: Broadcast[scala.collection.Map[(Int, Int), Int]],
                       coreUserFriendCountBC: Broadcast[scala.collection.Map[Int, Int]]) : RDD[((Int, Int), (Vector, Double))] = {
        pairScores
            .map(pair => {
                val features = createFeatures(pair, ageSexBC, cityPairCountBC, coreUserFriendCountBC)
                (pair.person1, pair.person2) -> features
            })
            .leftOuterJoin(positives)
            .map(t => {
                val pairUids = t._1
                val features = t._2._1
                val areFriends = t._2._2.getOrElse(0.0)

                pairUids -> (features, areFriends)
            })
    }

    def readFeatures(sqlc: SQLContext,
                     dataPath: String) : RDD[((Int, Int), (Vector, Double))] = {
        sqlc.read.parquet(Paths.getFeaturesPath(dataPath))
            .map(x => (x.getAs[Row](0), x.getAs[Row](1)))
            .map(
                x => (x._1.getAs[Int](0), x._1.getAs[Int](1)) -> (x._2.getAs[Vector](0), x._2.getAs[Double](1))
            )
    }

    def savePairFeatureLabelsToText(data: RDD[((Int, Int), (Vector, Double))],
                                    outPath: String,
                                    numPartitions: Int) : Unit = {
        data.map(entry => {
            val pair = entry._1
            val features = entry._2._1
            val label = entry._2._2
            val person1 = pair._1
            val person2 = pair._2
            val lineBuilder = Array.newBuilder[AnyVal]
            lineBuilder += person1
            lineBuilder += person2
            lineBuilder += label
            features.foreachActive((i, value) => {
                lineBuilder += value
            })
            lineBuilder.result().mkString(", ")
        }).repartition(numPartitions).saveAsTextFile(outPath)
    }

    /**
      * (Model, Threshold)
      */
    def trainModel(   sc : SparkContext,
                      trainData: RDD[((Int, Int), (Vector, Double))],
                      dataDir: String) : (classification.LogisticRegressionModel, Double) = {
        val allTrainData = trainData.map(t => LabeledPoint(t._2._2, t._2._1))

        val positiveData = allTrainData
            .filter(labeledPoint => labeledPoint.label > 0.9)
        val negativeData = allTrainData
            .filter(labeledPoint => labeledPoint.label < 0.1)
        val allCount = allTrainData.count()
        val positiveCount = positiveData.count()
        val negativeCount = negativeData.count()

        val positiveFraction = 0.6
        val requiredNegativeSampleCount = Math.min(negativeCount.toInt,
            (positiveCount.toDouble * (1.0 / positiveFraction - 1.0)).toInt)
        val negativeSampleData = sc.parallelize(negativeData.takeSample(
            withReplacement = false,
            num = requiredNegativeSampleCount,
            seed = 155L
        ))
        val effectiveNegativeSampleCount = negativeSampleData.count
        val negativePercent = 100 * effectiveNegativeSampleCount / (effectiveNegativeSampleCount + positiveCount)

        // save counter to text file for debugging
        sc.parallelize(Seq[Long] (
                allCount,
                positiveCount,
                negativeCount,
                requiredNegativeSampleCount,
                effectiveNegativeSampleCount,
                negativePercent),
            1)
            .saveAsTextFile(dataDir + "trainData_stats")

        val data = positiveData.union(negativeSampleData).repartition(24)

        data.saveAsTextFile(Paths.getTrainDataPath(dataDir))

        // split data into training (10%) and validation (90%)
        // step 6
        val splits = data.randomSplit(Array(0.1, 0.9), seed = 11L)
        val training = splits(0).cache()
        val validation = splits(1)

        // run training algorithm to build the model
        val model = {
            new LogisticRegressionWithLBFGS()
                .setNumClasses(2)
                .run(training)
        }

        model.clearThreshold()

        model.save(sc, Paths.getModelPath(dataDir))

        val predictionAndLabels = {
            validation.map { case LabeledPoint(label, features) =>
                val prediction = model.predict(features)
                (prediction, label)
            }
        }

        // estimate model quality
        @transient val metricsLogReg = new BinaryClassificationMetrics(predictionAndLabels, 100)
        val threshold = metricsLogReg.fMeasureByThreshold(2.0).sortBy(-_._2).take(1)(0)._1

        val rocLogReg = metricsLogReg.areaUnderROC()
        println("model ROC = " + rocLogReg.toString)
        sc.parallelize(Array(rocLogReg), 1).saveAsTextFile(dataDir + "modelLog")

        (model, threshold)
    }

}
