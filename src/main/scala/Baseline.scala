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
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import scala.collection.mutable.ArrayBuffer

case class PairWithScore(
        person1: Int,
        person2: Int,
        commonFriendsCount: Int,
        adamAdarScore: Double,
        commonSchool: Int,
        commonWork: Int,
        commonCircle: Int,
        maskOr: Int,
        maskAnd: Int
)

case class UserFriends(user: Int, friends: Array[Int])

case class AgeSexCity(age: Int, sex: Int, city: Int)

case class RelationCount(count: Array[Int]) {
    override def toString: String = {
        val sb = StringBuilder.newBuilder
        sb.append("(")
        for (i <- 0 until count.length) {
            if (i > 0) {
                sb.append(", ")
            }
            sb.append(getRelationName(i)).append(": ").append(count(i))
        }
        sb.append(")")
        sb.toString()
    }

    def getRelationName(i: Int): String = {
        i match {
            case 1 => "Love"
            case 2 => "Spouse"
            case 3 => "Parent"
            case 4 => "Child"
            case 5 => "Brother/Sister"
            case 6 => "Uncle/Aunt"
            case 7 => "Relative"
            case 8 => "Close friend"
            case 9 => "Colleague"
            case 10 => "Schoolmate"
            case 11 => "Nephew"
            case 12 => "Grandparent"
            case 13 => "Grandchild"
            case 14 => "College/University fellow"
            case 15 => "Army fellow"
            case 16 => "Parent in law"
            case 17 => "Child in law"
            case 18 => "Godparent"
            case 19 => "Godchild"
            case 20 => "Playing together"
            case other => "Unknown(" + other + ")"
        }
    }
}

case class FriendMask(uid: Int, mask: Int)

case class UserFriendsMask(user: Int, friends: Array[FriendMask])

case class FriendStat(intCount: Int, extCount: Int)

object Baseline {

    def main(args: Array[String]) {

        val sparkConf = new SparkConf().setAppName("Baseline")
        val sc = new SparkContext(sparkConf)
        val sqlc = new SQLContext(sc)

        import sqlc.implicits._

        val dataDir = if (args.length >= 1) args(0) else "./"

        val STAGE_REVERSE = 1
        val STAGE_NEIGHBORS = 2
        val STAGE_PAIRS = 3
        val STAGE_COUNT_CITIES = 4
        val STAGE_PREPARE = 5
        val STAGE_TRAIN = 6

        val REL_SCHOOL = 10
        val REL_WORK = 9
        val REL_UNIVERSITY = 14
        val REL_ARMY = 15
        val MASK_SCHOOL = 1 << REL_SCHOOL
        val MASK_WORK = 1 << REL_WORK
        val MASK_UNIVERSITY = 1 << REL_UNIVERSITY
        val MASK_ARMY = 1 << REL_ARMY

        var stage = 0
        var fromPart = 33
        var toPart = 42

        def parseArgs(): Unit = {
            for (i <- 1 until args.length) {
                if ("--stage".equals(args(i)) && i + 1 < args.length) {
                    val stageName = args(i + 1)
                    if ("reverse".equals(stageName)) {
                        stage = STAGE_REVERSE
                    } else if ("pairs".equals(stageName)) {
                        stage = STAGE_PAIRS
                    } else if ("cities".equals(stageName)) {
                        stage = STAGE_COUNT_CITIES
                    } else if ("prepare".equals(stageName)) {
                        stage = STAGE_PREPARE
                    } else if ("train".equals(stageName)) {
                        stage = STAGE_TRAIN
                    } else if ("neighbors".equals(stageName)) {
                        stage = STAGE_NEIGHBORS
                    }

                } else if ("--fromPart".equals(args(i)) && i + 1 < args.length) {
                    fromPart = args(i + 1).toInt

                } else if ("--toPart".equals(args(i)) && i + 1 < args.length) {
                    toPart = args(i + 1).toInt

                } else if ("--part".equals(args(i)) && i + 1 < args.length) {
                    fromPart = args(i + 1).toInt
                    toPart = fromPart
                }
            }
        }

        parseArgs()

        val graphPath = dataDir + "trainGraph"
        val reversedGraphPath = dataDir + "trainSubReversedGraph"
        val reversedGraphTxtPath = dataDir + "trainSubReversedGraph_txt"
        val commonFriendsPath = dataDir + "commonFriendsPartitioned"
        val commonFriendsTextPath = dataDir + "commonFriendsPartitioned_txt"
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
        val relationsTxtPath = dataDir + "relations_txt"

        val checkGraph = false
        val checkGraphTxtPath = dataDir + "checkGraph_txt"
        val findMissingEdges = false
        val missingEdgesTxtPath = dataDir + "missingEdges_txt"
        val countRelations = false
        val countFriendStat = true
        val friendsStatTxtPath = dataDir + "friendsStat_txt"

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

        val graphMask = {
            sc.textFile(graphPath)
                .map(line => {
                    val lineSplit = line.split("\t")
                    val user = lineSplit(0).toInt
                    val friends = {
                        lineSplit(1)
                            .replace("{(", "")
                            .replace(")}", "")
                            .split("\\),\\(")
                            .map(t => {
                                val friendSplit = t.split(",")
                                FriendMask(friendSplit(0).toInt, friendSplit(1).toInt)
                            })
                    }
                    UserFriendsMask(user, friends)
                })
        }

        if (countRelations) {
            val relationCount = graphMask.flatMap(uf => uf.friends.map(friend => {
                val counts = Array.fill[Int](32)(0)
                var mask = 1;
                for (i <- 0 until 32) {
                    if ((friend.mask & mask) != 0) {
                        counts(i) = 1
                    }
                    mask <<= 1
                }
                RelationCount(counts)
            }))
            .reduce((rcount1, rcount2) => {
                for (i <- 0 until Math.min(rcount1.count.length, rcount2.count.length)) {
                    rcount1.count(i) = rcount1.count(i) + rcount2.count(i)
                }
                rcount1
            })
            sc.parallelize(Array(relationCount), 1).saveAsTextFile(relationsTxtPath)
        }

        if (countFriendStat) {
            val mainUsersBC = sc.broadcast(graph.map(uf => uf.user).collect().toSet)
            val friendStats = graph.map(uf => {
                var intCount = 0
                var extCount = 0
                for (uid <- uf.friends) {
                    if (mainUsersBC.value.contains(uid)) {
                        intCount = intCount + 1
                    } else {
                        extCount = extCount + 1
                    }
                }
                FriendStat(intCount, extCount)
            }).reduce((stat1, stat2) => FriendStat(stat1.intCount + stat2.intCount, stat1.extCount + stat2.extCount))

            sc.parallelize(friendStats, 1).saveAsTextFile(friendsStatTxtPath)
            return
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

        def getReversedMaskGraph(minEdges: Int, maxEdges: Int, minReversedEdges: Int, maxReversedEdges: Int) = {
            graphMask
                .filter(userFriends =>
                        userFriends.friends.length >= minEdges
                        && userFriends.friends.length <= maxEdges)
                .flatMap(userFriends => userFriends.friends.map(x => (x.uid, FriendMask(userFriends.user, x.mask))))
                .groupByKey(numPartitions)
                .map(t => UserFriendsMask(t._1, t._2.toArray.sortWith((fm1, fm2) => fm1.uid < fm2.uid)))
                .filter(userFriends =>
                        userFriends.friends.length >= minReversedEdges
                        && userFriends.friends.length <= maxReversedEdges)
                .map(userFriends => (userFriends.user, userFriends.friends.map(fm => (fm.uid, fm.mask))))
        }


        if (stage <= STAGE_REVERSE) {
            val reversedGraph = getReversedMaskGraph(1, 2000, 2, 1000)

            reversedGraph
                .toDF
                .write.parquet(reversedGraphPath)

            reversedGraph.map(t => (t._1, t._2.toVector)).repartition(16).saveAsTextFile(reversedGraphTxtPath)
        }

        if (stage <= STAGE_NEIGHBORS) {
            val mainUsers = graph.map(userFriends => userFriends.user)
            val mainUsersBC = sc.broadcast(mainUsers.collect().toSet)

            val mainNeighborsCount = graph.map(userFriends => (userFriends.user, userFriends.friends.length))

            val reversedGraphRDD = sqlc.read.parquet(reversedGraphPath)
            reversedGraphRDD.printSchema()
            val reversedGraph = reversedGraphRDD
                    .map((a: Row) => (a.getAs[Int](0), a.getAs[Seq[Row]](1).map{case Row(f: Int, m: Int) => FriendMask(f, m)}.toArray))
                    .map(t => UserFriendsMask(t._1, t._2))
            //reversedGraph.map(uf => (uf.user, uf.friends.toVector)).repartition(1).saveAsTextFile(reversedGraphTxtPath + "_restored")

            val otherNeighborsCount = reversedGraph
                .filter(uf => !mainUsersBC.value.contains(uf.user))
                .map(uf => (uf.user, uf.friends.length))

            val neighborsCount = mainNeighborsCount.union(otherNeighborsCount)
            neighborsCount.map(t => t.swap).sortByKey(ascending = false).repartition(16).saveAsTextFile(neighborsCountTxtPath)
            neighborsCount.toDF.repartition(16).write.parquet(neighborsCountPath)
         }

        if (stage <= STAGE_PAIRS) {
            val neighborsCount = sqlc.read.parquet(neighborsCountPath)
                    .map(t => (t.getAs[Int](0), t.getAs[Int](1)))

            val neighborsCountBC = sc.broadcast(neighborsCount.collectAsMap())

            def genAdamAdarScore(uf: UserFriendsMask, numPartitions: Int, k: Int) = {
                // person1, person2 -> common friends, adam adair score, common school, common work
                val pairs = ArrayBuffer.empty[((Int, Int), (Int, Double, Int, Int, Int, Int, Int))]

                // get Adam-Adair score for the common friend
                val nCount = neighborsCountBC.value.getOrElse(uf.user, 0)
                val score = if (nCount >= 2) {
                    1.0 / Math.log(nCount.toDouble)
                } else {
                    0.0
                }
                for (i <- 0 until uf.friends.length) {
                    val p1 = uf.friends(i).uid
                    val mask1 = uf.friends(i).mask
                    val school1 = if ((mask1 & MASK_SCHOOL) != 0) 1 else 0
                    val work1 = if ((mask1 & MASK_WORK) != 0) 1 else 0
                    val univ1 = if ((mask1 & MASK_UNIVERSITY) != 0) 1 else 0
                    val army1 = if ((mask1 & MASK_ARMY) != 0) 1 else 0

                    if (p1 % numPartitions == k) {
                        for (j <- i + 1 until uf.friends.length) {
                            val p2 = uf.friends(j).uid
                            val mask2 = uf.friends(j).mask
                            val school2 = if ((mask2 & MASK_SCHOOL) != 0) 1 else 0
                            val work2 = if ((mask2 & MASK_WORK) != 0) 1 else 0
                            val univ2 = if ((mask2 & MASK_UNIVERSITY) != 0) 1 else 0
                            val army2 = if ((mask2 & MASK_ARMY) != 0) 1 else 0

                            val commonCircle = if (work1 * work2 + school1 * school2 + univ1 * univ2 + army1 * army2 > 0) 1 else 0

                            pairs.append(((p1, p2), (1, score, school1 * school2, work1 * work2, commonCircle, mask1 | mask2, mask1 & mask2)))
                        }
                    }
                }
                pairs
            }
            for (k <- 0 until numPartitionsGraph) {
                val commonFriendPairs = {
                    sqlc.read.parquet(reversedGraphPath)
                        .map((a: Row) => (a.getAs[Int](0), a.getAs[Seq[Row]](1).map{case Row(f: Int, m: Int) => FriendMask(f, m)}.toArray))
                        .map(t => UserFriendsMask(t._1, t._2))
                        .flatMap(t => genAdamAdarScore(t, numPartitionsGraph, k))
                        .reduceByKey((val1, val2) => (
                            val1._1 + val2._1,
                            val1._2 + val2._2,
                            val1._3 + val2._3,
                            val1._4 + val2._4,
                            val1._5 + val2._5,
                            val1._6 | val2._6,
                            val1._7 | val2._7))
                        .map(t => PairWithScore(t._1._1, t._1._2, t._2._1, t._2._2, t._2._3, t._2._4, t._2._5, t._2._6, t._2._7))
                        .filter(pair => pair.adamAdarScore >= 1.0)
                }

                commonFriendPairs.repartition(4).toDF.write.parquet(commonFriendsPath + "/part_" + k)
                if (k == 0) {
                    commonFriendPairs.repartition(1).map(pair =>
                            (pair.adamAdarScore, (pair.person1, pair.person2, pair.commonFriendsCount, pair.commonSchool, pair.commonWork, pair.commonCircle, pair.maskOr, pair.maskAnd)))
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

            cityPairCount.map(t => (t._1._1, t._1._2, t._2)).repartition(16).toDF().write.parquet(cityPairCountPath)
            cityPairCount.map(t => t.swap).sortByKey(ascending = false).saveAsTextFile(cityPairCountTxtPath)
        }

        val cityPairCount = {
            sqlc.read.parquet(cityPairCountPath)
                .map(t => ((t.getAs[Int](0), t.getAs[Int](1)), t.getAs[Int](2)))
        }

        val cityPairCountBC = sc.broadcast(cityPairCount.collectAsMap())

        var adamAdairPairs = sc.parallelize(Seq[PairWithScore] ())

        for (partNo <- fromPart until toPart + 1) {
            // prepare data for training model
            // step 2
            val pairsPart = {
                sqlc
                    .read.parquet(commonFriendsPath + "/part_" + partNo)
                    .map(t => PairWithScore(t.getAs[Int](0), t.getAs[Int](1),
                        t.getAs[Int](2), t.getAs[Double](3), t.getAs[Int](4), t.getAs[Int](5),
                        t.getAs[Int](6), t.getAs[Int](7), t.getAs[Int](8)))
            }
            adamAdairPairs = adamAdairPairs.union(pairsPart)
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
                        if (cityCount > 50000) {
                            0.5
                        } else {
                            0.0
                        }
                    }
                    val diffAge = abs(ageSexCity1.age - ageSexCity2.age).toDouble
                    val meanAge = 25000.0 - (ageSexCity1.age + ageSexCity2.age) * 0.5

                    Vectors.dense(
                        Math.log(pair.adamAdarScore),
                        Math.log(diffAge + 1.0),
                        Math.log(meanAge),
                        if (isSameSex) 1.0 else 0.0,
                        cityFactor,
                        pair.commonSchool,
                        pair.commonWork,
                        pair.commonCircle,
                        if (pair.maskAnd > 1) 1.0 else 0.0
//                        if (pair.maskOr > 1) 1.0 else 0.0
                    )
                })
                .leftOuterJoin(positives)
        }

        val allPairsFeatures = prepareData(adamAdairPairs, positives, ageSexCityBC, cityPairCountBC)
        allPairsFeatures.map(pairToFeaturesJoinPositives => {
            val pair = pairToFeaturesJoinPositives._1
            val joinedValue = pairToFeaturesJoinPositives._2
            val features = joinedValue._1
            val positiveOpt = joinedValue._2
            val positive = if (positiveOpt.isDefined) 1 else 0
            val person1 = pair._1
            val person2 = pair._2
            val lineBuilder = Array.newBuilder[AnyVal]
            lineBuilder += person1
            lineBuilder += person2
            lineBuilder += positive
            features.foreachActive((i, value) => { lineBuilder += value })
            lineBuilder.result().mkString(", ")
        }).repartition(1).saveAsTextFile(dataDir + "allPairsData_txt")


        val allData = allPairsFeatures.map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))

        val positiveData = allData
            .filter(labeledPoint => labeledPoint.label > 0.9)
        val negativeData = allData
            .filter(labeledPoint => labeledPoint.label < 0.1)
        val allCount = allData.count()
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

        sc.parallelize(Seq[Long](allCount, positiveCount, negativeCount,
                                 requiredNegativeSampleCount, effectiveNegativeSampleCount, negativePercent), 1)
                    .saveAsTextFile(dataDir + "trainData_stats")

        val data = positiveData.union(negativeSampleData).repartition(24)

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
        sc.parallelize(Array(rocLogReg), 1).saveAsTextFile(dataDir + "modelLog")


        // compute scores on the test set
        // step 7
        val testCommonFriendsCounts = {
            sqlc
                .read.parquet(commonFriendsPath + "/part_*/")
                .map(t => PairWithScore(t.getAs[Int](0), t.getAs[Int](1),
                    t.getAs[Int](2), t.getAs[Double](3), t.getAs[Int](4), t.getAs[Int](5),
                    t.getAs[Int](6), t.getAs[Int](7), t.getAs[Int](8)))
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
                    val topBestFriends = friendsWithRatings.toList.sortBy(-_._2).take(70).map(x => x._1)
                    (user, topBestFriends)
                })
                .sortByKey(true, 1)
                .map(t => t._1 + "\t" + t._2.mkString("\t"))
        }

        testPrediction.saveAsTextFile(predictionPath, classOf[GzipCodec])
    }
}