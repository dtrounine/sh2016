/**
  * Baseline for hackaton
  */


import breeze.numerics.abs
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import scala.collection.mutable.ArrayBuffer

case class PairWithCommonFriends(person1: Int, person2: Int, commonFriendsCount: Int)

case class PairWithScore(person1: Int, person2: Int, score: Double)

case class UserFriends(user: Int, friends: Array[Int])

case class AgeSexCity(age: Int, sex: Int, city: Int)

object Baseline {

    def main(args: Array[String]) {

        val sparkConf = new SparkConf().setAppName("Baseline")
        val sc = new SparkContext(sparkConf)
        val sqlc = new SQLContext(sc)

        import sqlc.implicits._

        val dataDir = if (args.length >= 1) args(0) else "./"

        val STAGE_REVERSE = 1
        val STAGE_COUNT_COMMON_FRIENDS = 2
        val STAGE_NEIGHBORS = 3
        val STAGE_COUNT_CITIES = 4
        val STAGE_PREPARE = 5
        val STAGE_TRAIN = 6

        def getStageFromArgs(): Int = {
            for (i <- 1 until args.length) {
                if ("--stage".equals(args(i)) && i + 1 < args.length) {
                    val stageName = args(i + 1)
                    if ("reverse".equals(stageName)) {
                        return STAGE_REVERSE
                    } else if ("count".equals(stageName)) {
                        return STAGE_COUNT_COMMON_FRIENDS
                    } else if ("cities".equals(stageName)) {
                        return STAGE_COUNT_CITIES
                    } else if ("prepare".equals(stageName)) {
                        return STAGE_PREPARE
                    } else if ("train".equals(stageName)) {
                        return STAGE_TRAIN
                    } else if ("neighbors".equals(stageName)) {
                        return STAGE_NEIGHBORS
                    }
                }
            }
            0
        }

        val stage = getStageFromArgs()

        val graphPath = dataDir + "trainGraph"
        val reversedGraphPath = dataDir + "trainSubReversedGraph"
        val reversedGraphTxtPath = dataDir + "trainSubReversedGraph_txt"
        val commonFriendsPath = dataDir + "commonFriendsCountsPartitioned"
        val commonFriendsTextPath = dataDir + "commonFriendsCountsPartitioned_txt"
        val demographyPath = dataDir + "demography"
        val predictionPath = dataDir + "prediction"
        val trainDataPath = dataDir + "trainData"
        val modelPath = dataDir + "LogisticRegressionModel"
        val cityPairCountPath = dataDir + "cityPairCount"
        val cityPairCountTxtPath = cityPairCountPath + "_txt"
        val cityPopulationPath = dataDir + "cityPopulation"
        val cityPopulationTxtPath = cityPopulationPath + "_txt"
        val neighborsCountPath = dataDir + "neighborsCount"
        val neighborsCountTxtPath = neighborsCountPath + "_txt"
        val adamAdairPath = dataDir + "adamAdair"
        val adamAdairTxtPath = adamAdairPath + "_txt"

        val checkGraph = false
        val checkGraphTxtPath = dataDir + "checkGraph_txt"
        val findMissingEdges = false
        val missingEdgesTxtPath = dataDir + "missingEdges_txt"

        val numPartitions = 200
        val numPartitionsGraph = 107
//        val numPartitions = 1
//        val numPartitionsGraph = 1

        // read graph, flat and reverse it
        // step 1.a from description

        val graph = {
            sc.textFile(graphPath)
                .map(line => {
                    val lineSplit = line.split("\t")
                    val user = lineSplit(0).toInt
                    val friends = {
                        lineSplit(1)
                            .replace("{(", "")
                            .replace(")}", "")
                            .split("\\),\\(")
                            .map(t => t.split(",")(0).toInt)
                    }
                    UserFriends(user, friends)
                })
        }

        if (checkGraph) {
            val mainUsersBC = sc.broadcast(graph.map(uf => uf.user).collect().toSet)

            val reversedGraph = graph
                .flatMap(userFriends => userFriends.friends.map(x => (x, userFriends.user)))
                .groupByKey(numPartitions)
                .filter(t => mainUsersBC.value.contains(t._1))
                .map(t => UserFriends(t._1, t._2.toArray))

            val nBC = sc.broadcast(graph
                .map(uf => (uf.user, uf.friends.filter(x => mainUsersBC.value.contains(x)).length))
                .collectAsMap())

            reversedGraph.filter(uf => uf.friends.length != nBC.value.getOrElse(uf.user, 0))
                .map(uf => (uf.user, uf.friends.toSeq))
                .repartition(1)
                .saveAsTextFile(checkGraphTxtPath)
            return
        }


        if (findMissingEdges) {
            val coreNodesBC = sc.broadcast(graph.map(node => node.user).collect().toSet)
            val coreGraph = graph
                .map(node => (
                    node.user,
                    node.friends.filter(x => coreNodesBC.value.contains(x))
                ))
            val reversedCoreGraph = coreGraph
                .flatMap(node => node._2.map(x => (x, node._1)))
                .groupByKey(numPartitions)

            // (node_id, out_edges, in_edges)
            val nodesOutAndIn =
                coreGraph.fullOuterJoin(reversedCoreGraph)
                    .map(x => (
                        x._1,
                        if (x._2._1.isDefined) x._2._1.get.toArray[Int].sorted else Array[Int](),
                        if (x._2._2.isDefined) x._2._2.get.toArray[Int].sorted else Array[Int]()
                    ))

            // input arrays must be sorted
            def arrayDiff(a: Array[Int], b: Array[Int]) = {
                val onlyInA = ArrayBuffer.empty[Int]
                val onlyInB = ArrayBuffer.empty[Int]

                var i = 0
                var j = 0
                while (i < a.length && j < b.length) {
                    if (a(i) == b(j)) {
                        i = i + 1
                        j = j + 1
                    } else if (a(i) < b(j)) {
                        onlyInA.append(a(i))
                        i = i + 1
                    } else {
                        onlyInB.append(b(j))
                        j = j + 1
                    }
                }
                while (i < a.length) {
                    onlyInA.append(a(i))
                    i = i + 1
                }
                while (j < b.length) {
                    onlyInB.append(b(j))
                    j = j + 1
                }
                (onlyInA.toArray, onlyInB.toArray)
            }

            // (node_id, missing_out, missing_in
            val missingEdges = nodesOutAndIn
                    .map(t => {
                        val diff = arrayDiff(t._2, t._3)
                        (t._1, diff._2, diff._1)
                    })
                    .filter(t => t._2.length > 0 || t._3.length > 0)

            missingEdges.map(t => (t._1, t._2.toVector, t._3.toVector)).repartition(16).saveAsTextFile(missingEdgesTxtPath)

            return
        }

        def getReversedGraph(minEdges: Int, maxEdges: Int, minReversedEdges: Int, maxReversedEdges: Int) = {
            graph
                .filter(userFriends =>
                        userFriends.friends.length >= minEdges
                        && userFriends.friends.length <= maxEdges)
                .flatMap(userFriends => userFriends.friends.map(x => (x, userFriends.user)))
                .groupByKey(numPartitions)
                .map(t => UserFriends(t._1, t._2.toArray.sorted))
                .filter(userFriends =>
                        userFriends.friends.length >= minReversedEdges
                        && userFriends.friends.length <= maxReversedEdges)
                .map(userFriends => Tuple2(userFriends.user, userFriends.friends))
        }


        if (stage <= STAGE_REVERSE) {
            getReversedGraph(1, 2000, 2, 1000)
                .toDF
                .write.parquet(reversedGraphPath)

//            reversedGraph.map(t => (t._1, t._2.toVector)).repartition(16).saveAsTextFile(reversedGraphTxtPath)

        }

        if (stage <= STAGE_NEIGHBORS) {
/*
            val mainUsers = graph.map(userFriends => userFriends.user)
            val mainUsersBC = sc.broadcast(mainUsers.collect().toSet)

            val mainNeighborsCount = graph.map(userFriends => (userFriends.user, userFriends.friends.length))

            val otherNeighborsCount = sqlc.read.parquet(reversedGraphPath)
                .map(t => (t.getAs[Int](0), t.getAs[Seq[Int]](1)))
                .filter(t => !mainUsersBC.value.contains(t._1))
                .map(t => (t._1, t._2.length))

            val neighborsCount = mainNeighborsCount.union(otherNeighborsCount)
            neighborsCount.map(t => t.swap).sortByKey(ascending = false).repartition(16).saveAsTextFile(neighborsCountTxtPath)
            neighborsCount.toDF.repartition(16).write.parquet(neighborsCountPath)
*/
            val neighborsCount = sqlc.read.parquet(neighborsCountPath)
                    .map(t => (t.getAs[Int](0), t.getAs[Int](1)))

            val neighborsCountBC = sc.broadcast(neighborsCount.collectAsMap())

            def genAdamAdarScore(user: Int, friends:Seq[Int], numPartitions: Int, k: Int) = {
                val pairs = ArrayBuffer.empty[((Int, Int), Double)]
                val nCount = neighborsCountBC.value.getOrElse(user, 0)
                val score = if (nCount >= 2) {
                    1.0 / Math.log(nCount.toDouble)
                } else {
                    0.0
                }
                if (score != 0.0) {
                    for (i <- 0 until friends.length) {
                        if (friends(i) % numPartitions == k) {
                            for (j <- i + 1 until friends.length) {
                                pairs.append(((friends(i), friends(j)), score))
                            }
                        }
                    }
                }
                pairs
            }

            for (k <- 0 until numPartitionsGraph) {
                val adamAdarPairs = {
                    sqlc.read.parquet(reversedGraphPath)
                        .map(t => (t.getAs[Int](0), t.getAs[Seq[Int]](1)))
                        .filter(t => t._2.length >= 2)
                        .flatMap(t => genAdamAdarScore(t._1, t._2, numPartitionsGraph, k))
                        .reduceByKey((x, y) => x + y)
                        .map(t => PairWithScore(t._1._1, t._1._2, t._2))
                        .filter(pair => pair.score >= 1.0)
                }

                adamAdarPairs.repartition(4).toDF.write.parquet(adamAdairPath + "/part_" + k)
                if (k == 0) {
                    adamAdarPairs.repartition(1).map(pair => (pair.score, (pair.person1, pair.person2)))
                        .sortByKey(ascending = false)
                        .saveAsTextFile(adamAdairTxtPath + "/part_" + k)
                }
            }

        }

        // for each pair of ppl count the amount of their common friends
        // amount of shared friends for pair (A, B) and for pair (B, A) is the same
        // so order pair: A < B and count common friends for pairs unique up to permutation
        // step 1.b

        def generatePairs(pplWithCommonFriends: Seq[Int], numPartitions: Int, k: Int) = {
            val pairs = ArrayBuffer.empty[(Int, Int)]
            for (i <- 0 until pplWithCommonFriends.length) {
                if (pplWithCommonFriends(i) % numPartitions == k) {
                    for (j <- i + 1 until pplWithCommonFriends.length) {
                        pairs.append((pplWithCommonFriends(i), pplWithCommonFriends(j)))
                    }
                }
            }
            pairs
        }

        if (stage <= STAGE_COUNT_COMMON_FRIENDS) {
            for (k <- 0 until numPartitionsGraph) {
                val commonFriendsCounts = {
                    sqlc.read.parquet(reversedGraphPath)
                        .map(t => t.getAs[Seq[Int]](1))
                        .map(t => generatePairs(t, numPartitionsGraph, k))
                        .flatMap(pair => pair.map(x => x -> 1))
                        .reduceByKey((x, y) => x + y)
                        .map(t => PairWithCommonFriends(t._1._1, t._1._2, t._2))
                        .filter(pair => pair.commonFriendsCount > 8)
                }

                commonFriendsCounts.repartition(4).toDF.write.parquet(commonFriendsPath + "/part_" + k)
                if (k == 0) {
                    commonFriendsCounts.repartition(1).map(pair => (pair.commonFriendsCount, (pair.person1, pair.person2)))
                        .sortByKey(ascending = false)
                        .saveAsTextFile(commonFriendsTextPath + "/part_" + k)
                }
            }
        }

        if (stage <= STAGE_COUNT_CITIES) {
            val userCity = {
                sc.textFile(demographyPath)
                    .map(line => {
                        val lineSplit = line.trim().split("\t")
                        lineSplit(0).toInt -> lineSplit(5).toInt
                    })
            }

            val cityPopulation =
                userCity
                    .map(t => t._2 -> 1)
                    .reduceByKey((x, y) => x + y)
            cityPopulation.map(t => t.swap).sortByKey(ascending = false).saveAsTextFile(cityPopulationTxtPath)

            val cityPopBC = sc.broadcast(cityPopulation.collectAsMap())
            val userCityBC = sc.broadcast(userCity.collectAsMap())

            val cityPairCount = graph.flatMap(userFriends => userFriends.friends.map(x => (x, userFriends.user)))
                .map(t => {
                    val user1 = t._1
                    val user2 = t._2
                    val city1 = userCityBC.value.getOrElse(user1, 0)
                    val city2 = userCityBC.value.getOrElse(user2, 0)
                    if (city1 < city2) (city1, city2) else (city2, city1)
                })
                .filter(t => t._1 > 0 && t._2 > 0)
                .map(x => x -> 1)
                .reduceByKey((count1, count2) => count1 + count2)
                .filter(t => t._2 >= 5)

            cityPairCount.map(t => (t._1._1, t._1._2, t._2)).repartition(4).toDF().write.parquet(cityPairCountPath)
            cityPairCount.map(t => t.swap).sortByKey(ascending = false).saveAsTextFile(cityPairCountTxtPath)

            return
        }

        val cityPairCount = {
            sqlc.read.parquet(cityPairCountPath)
                .map(t => ((t.getAs[Int](0), t.getAs[Int](1)), t.getAs[Int](2)))
        }

        val cityPairCountBC = sc.broadcast(cityPairCount.collectAsMap())

        // prepare data for training model
        // step 2

        val commonFriendsCounts = {
            sqlc
                .read.parquet(commonFriendsPath + "/part_33")
                .map(t => PairWithCommonFriends(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Int](2)))
        }

        val adamAdairPairs = {
            sqlc
                    .read.parquet(adamAdairPath + "/part_33")
                    .map(t => PairWithScore(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Double](2)))
        }

        // step 3
        val usersBC = sc.broadcast(graph.map(userFriends => userFriends.user).collect().toSet)

        val positives = {
            graph
                .flatMap(
                    userFriends => userFriends.friends
                        .filter(x => (usersBC.value.contains(x) && x > userFriends.user))
                        .map(x => (userFriends.user, x) -> 1.0)
                )
        }

        // step 4
        val ageSexCity = {
            sc.textFile(demographyPath)
                .map(line => {
                    val lineSplit = line.trim().split("\t")
                    if (lineSplit(2) == "") {
                        (lineSplit(0).toInt -> AgeSexCity(0, lineSplit(3).toInt, lineSplit(5).toInt))
                    }
                    else {
                        (lineSplit(0).toInt -> AgeSexCity(lineSplit(2).toInt, lineSplit(3).toInt, lineSplit(5).toInt))
                    }
                })
        }

        val ageSexCityBC = sc.broadcast(ageSexCity.collectAsMap())

        // step 5
        def prepareData(
                           //commonFriendsCounts: RDD[PairWithCommonFriends],
                           adamAdairScores: RDD[PairWithScore],
                           positives: RDD[((Int, Int), Double)],
                           ageSexBC: Broadcast[scala.collection.Map[Int, AgeSexCity]],
                           cityPairCountBS: Broadcast[scala.collection.Map[(Int, Int), Int]]) = {

            //commonFriendsCounts
            adamAdairScores
                //.map(pair => (pair.person1, pair.person2) -> {
                .map(pair => (pair.person1, pair.person2) -> {
                    val ageSexCity1 = ageSexBC.value.getOrElse(pair.person1, AgeSexCity(0, 0, 0))
                    val ageSexCity2 = ageSexBC.value.getOrElse(pair.person2, AgeSexCity(0, 0, 0))
                    val isSameSex = ageSexCity1.sex == ageSexCity2.sex
                    val city1 = ageSexCity1.city
                    val city2 = ageSexCity2.city
                    val cityFactor = if (city1 == city2) {
                        1.0
                    } else {
                        val cityPair = if (city1 < city2) (city1, city2) else (city2, city1)
                        val cityCount = cityPairCountBS.value.getOrElse(cityPair, 0)
                        if (cityCount > 1000) {
                            0.5
                        } else {
                            0.0
                        }
                    }

                    Vectors.dense(
                        //pair.commonFriendsCount.toDouble,
                        Math.log(pair.score),
                        abs(ageSexCity1.age - ageSexCity2.age).toDouble,
                        (ageSexCity1.age + ageSexCity2.age) * 0.5,
                        if (isSameSex) 1.0 else 0.0,
                        cityFactor
                    )
                })
                .leftOuterJoin(positives)
        }

//        val data = {
//            prepareData(commonFriendsCounts, positives, ageSexCityBC, cityPairCountBC)
//                .map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
//        }
        val data = {
            prepareData(adamAdairPairs, positives, ageSexCityBC, cityPairCountBC)
                .map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
        }

        data.saveAsTextFile(trainDataPath)

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
        model.save(sc, modelPath)

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
        sc.parallelize(Array(rocLogReg)).saveAsTextFile(dataDir + "modelLog")


        // compute scores on the test set
        // step 7
//        val testCommonFriendsCounts = {
//            sqlc
//                .read.parquet(commonFriendsPath + "/part_*/")
//                .map(t => PairWithCommonFriends(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Int](2)))
//                .filter(pair => pair.person1 % 11 == 7 || pair.person2 % 11 == 7)
//        }

        // compute scores on the test set
        // step 7
        val testCommonFriendsCounts = {
            sqlc
                .read.parquet(adamAdairPath + "/part_*/")
                .map(t => PairWithScore(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Double](2)))
                .filter(pair => pair.person1 % 11 == 7 || pair.person2 % 11 == 7)
        }

        val testData = {
            prepareData(testCommonFriendsCounts, positives, ageSexCityBC, cityPairCountBC)
                .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
                .filter(t => t._2.label == 0.0)
        }

        // step 8
        val testPrediction = {
            testData
                .flatMap { case (id, LabeledPoint(label, features)) =>
                    val prediction = model.predict(features)
                    Seq(id._1 ->(id._2, prediction), id._2 ->(id._1, prediction))
                }
                .filter(t => t._1 % 11 == 7 && t._2._2 >= threshold)
                .groupByKey(numPartitions)
                .map(t => {
                    val user = t._1
                    val friendsWithRatings = t._2
                    val topBestFriends = friendsWithRatings.toList.sortBy(-_._2).take(100).map(x => x._1)
                    (user, topBestFriends)
                })
                .sortByKey(true, 1)
                .map(t => t._1 + "\t" + t._2.mkString("\t"))
        }

        testPrediction.saveAsTextFile(predictionPath, classOf[GzipCodec])
    }
}