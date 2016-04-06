/**
  * Baseline for hackaton
  */


import breeze.numerics.abs
import org.apache.hadoop.fs.Path
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
import org.apache.spark.graphx._

import scala.collection.mutable.ArrayBuffer

/**
 * Пара пользователей со счетчиком общих друзей и другими счетчиками,
 * которые вычисляются одновременно с ним
 */
case class PairWithScore(
        person1: Int,
        person2: Int,
        commonFriendsCount: Int,
        aaScore: Double,
        fedorScore: Double,
        interactionScore: Double,
        pageRankScore: Double,
        commonSchool: Int,
        commonWork: Int,
        commonArmy: Int,
        commonUniversity: Int,
        maskOr: Int,
        maskAnd: Int
)

case class UserFriends(user: Int, friends: Array[Int])

case class AgeSexCity(age: Int, sex: Int, city: Int)

case class UserPageRank(user: Long, rank: Double)

case class FriendMask(uid: Int, mask: Int)

case class UserFriendsMask(user: Int, friends: Array[FriendMask])

case class FriendStat(intCount: Int, extCount: Int)

case class Interaction(from: Long, to: Long, entries: Seq[(Int, Double)])

object Baseline {

    def main(args: Array[String]) {

        val sparkConf = new SparkConf().setAppName("Baseline")
        val sc = new SparkContext(sparkConf)
        val sqlc = new SQLContext(sc)
        val conf = sc.hadoopConfiguration
        val fs = org.apache.hadoop.fs.FileSystem.get(conf)

        import sqlc.implicits._

        val dataDir = if (args.length >= 1) args(0) else "./"

        val STAGE_REVERSE = 1
        val STAGE_PAGE_RANK = 2
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
                    } else if ("pageRank".equals(stageName)) {
                        stage = STAGE_PAGE_RANK
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
        val userPageRankPath = dataDir + "userPageRank"
        val userPageRankTxtPath = userPageRankPath + "_txt"
        val otherDetailsPath = dataDir + "otherDetails"
        val interactionsPath = dataDir + "interactions"

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

        val interactions = {
            sqlc.read.parquet(interactionsPath)
                .map{ case Row(from: Long, to: Long, entries: Seq[(Int, Double)]) => Interaction(from, to, entries) }
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


        if (stage <= STAGE_REVERSE && !fs.exists(new Path(reversedGraphPath))) {
            val reversedGraph = getReversedMaskGraph(1, 2000, 2, 1000)

            reversedGraph
                .toDF
                .write.parquet(reversedGraphPath)

            reversedGraph.map(t => (t._1, t._2.toVector)).repartition(16).saveAsTextFile(reversedGraphTxtPath)
        }

        val mainUsers = graph.map(userFriends => userFriends.user)
        val mainUsersBC = sc.broadcast(mainUsers.collect().toSet)

        val otherDetails = sqlc.read.parquet(otherDetailsPath)


        if (stage <= STAGE_PAGE_RANK && !fs.exists(new Path(userPageRankPath))) {
            def computePageRank(kernelUsers: Broadcast[Set[Int]]) = {
                val numIterations = 5
                val edges = {
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
                            }.filter(f => kernelUsers.value.contains(f))
                            (user, friends)
                        }).flatMap(uf => uf._2.map(x => Edge(uf._1: VertexId, x: VertexId, 1)))
                }

                Graph.fromEdges(edges, 1).staticPageRank(numIterations).vertices
                    .map(v => UserPageRank(v._1, v._2))
                    .toDF()
                    .write.parquet(userPageRankPath)
            }

            computePageRank(mainUsersBC)
        }

        def useForTraining(uid: Int): Boolean = {
            val part = uid % numPartitionsGraph
            part >= fromPart || part <= toPart
        }

        def useForPrediction(uid: Int): Boolean = {
            uid % 11 == 7
        }

        def useUser(uid: Int): Boolean = {
             useForTraining(uid) || useForPrediction(uid)
        }

        if (stage <= STAGE_PAIRS && !fs.exists(new Path(commonFriendsPath))) {

            val pageRank = {
                sqlc.read.parquet(userPageRankPath)
                    .map{case Row(k: Long, v: Double) => k.toInt -> v}

            }
            val maxPageRank = pageRank.map(t => t._2).max()
            val normalizedPageRank = pageRank.map(t => t._1 -> t._2 / maxPageRank)
            val normalizedPageRankBC = sc.broadcast(normalizedPageRank.collectAsMap())

            otherDetails.printSchema()

            val otherNeighborsCount = otherDetails.map(t => (
                t.getAs[Long](0), // UID
                t.getAs[Long](1), // CreateDate
                t.getAs[Int](2), // BirthDate
                t.getAs[Int](3), // gender
                t.getAs[Long](4), // country ID
                t.getAs[Int](5), // location ID
                t.getAs[Int](6), // login region
                t.getAs[Int](7), // is active
                t.getAs[Long](8) // num friends
                ))
                .map(t => (t._1.toInt, t._9.toInt))

            val mainNeighborsCount = graph.map(userFriends => (userFriends.user, userFriends.friends.length))
            val neighborsCount = mainNeighborsCount.union(otherNeighborsCount)
            val neighborsCountBC = sc.broadcast(neighborsCount.collectAsMap())

            /*
            val interactions = {
                sqlc.read.parquet(interactionsPath)
                    .map{ case Row(from: Long, to: Long, entries: Seq[(Int, Double)]) => (from, to) -> entries }
            }
            val interactionsBC = sc.broadcast(interactions.collectAsMap())

            def getInteractionScore(interactionEntries: Seq[(Int, Double)]): Unit = {
                 for (interaction <- interactionEntries) {
                     val iType = interaction._1
                     val importance =
                         if (iType == 1) 0.0 // удаление фида из ленты
                         else if (iType == 2) 1.0 // поход в гости
                         else if (iType == 3) 1.0 // участие в опросе
                         else if (iType == 4) 1.0 // отправка личного сообщения
                         else if (iType == 5) 1.0 // удаление личного сообщения
                         else if (iType == 6) 1.0 // класс объекта
                         else if (iType == 7) 1.0 // разкласс объекта
                         else if (iType == 8) 1.0 // комментирование пользовательского поста
                         else if (iType == 9) 1.0 // комментирования пользовательского фото
                         else if (iType == 10) 1.0 // комментирование пользовательского видео
                         else if (iType == 11) 1.0 // комментирование фотоальбома
                         else if (iType == 12) 1.0 // класс к комментарию
                         else if (iType == 13) 1.0 // отправка сообщения на форуме
                         else if (iType == 14) 1.0 // оценка фото
                         else if (iType == 15) 1.0 // просмотр фото
                         else if (iType == 16) 10.0 // отметка пользователя на фотографиях
                         else if (iType == 17) 10.0 // отметка пользователя на отдельном фото
                         else if (iType == 18) 10.0 // отправка подарка
                         else 0.0
                 }
            }
*/
            def genPairScores(uf: UserFriendsMask, numPartitions: Int, k: Int) = {
                // person1, person2 -> common friends, adam adair score, common school, common work
                val pairs = ArrayBuffer.empty[((Int, Int), (Int, Double, Double, Double, Double, Int, Int, Int, Int, Int, Int))]

                // get Adam-Adair score for the common friend
                val nCount = neighborsCountBC.value.getOrElse(uf.user, 0)
                val aaScore = if (nCount >= 2) 1.0 / Math.log(nCount.toDouble) else 0.0
                val fedorScore = 100.0 / Math.pow(nCount.toDouble + 10, 1.0/3.0) - 6
                val pageRankScore = normalizedPageRankBC.value.getOrElse(uf.user, 0.0)

                for (i <- 0 until uf.friends.length) {
                    val p1 = uf.friends(i).uid

                    if (p1 % numPartitions == k) {

                        val mask1 = uf.friends(i).mask
                        val school1 = if ((mask1 & MASK_SCHOOL) != 0) 1 else 0
                        val work1 = if ((mask1 & MASK_WORK) != 0) 1 else 0
                        val univ1 = if ((mask1 & MASK_UNIVERSITY) != 0) 1 else 0
                        val army1 = if ((mask1 & MASK_ARMY) != 0) 1 else 0
                        val interaction1 = 1.0


                        for (j <- i + 1 until uf.friends.length) {
                            val p2 = uf.friends(j).uid

                            if (useUser(p1) || useUser(p2)) {

                                val mask2 = uf.friends(j).mask
                                val school2 = if ((mask2 & MASK_SCHOOL) != 0) 1 else 0
                                val work2 = if ((mask2 & MASK_WORK) != 0) 1 else 0
                                val univ2 = if ((mask2 & MASK_UNIVERSITY) != 0) 1 else 0
                                val army2 = if ((mask2 & MASK_ARMY) != 0) 1 else 0
                                val interaction2 = 1.0

                                val interactionScore = interaction1 * interaction2

                                pairs.append(((p1, p2), (
                                    1,
                                    aaScore,
                                    fedorScore,
                                    interactionScore,
                                    pageRankScore,
                                    school1 * school2,
                                    work1 * work2,
                                    army1 * army2,
                                    univ1 * univ2,
                                    mask1 | mask2,
                                    mask1 & mask2)))
                            }
                        }
                    }
                }
                pairs
            }
            for (k <- 0 until numPartitionsGraph) {
                val commonFriendPairs = {
                    sqlc.read.parquet(reversedGraphPath)
                        .map((a: Row) => (a.getAs[Int](0), a.getAs[Seq[Row]](1).map { case Row(f: Int, m: Int) => FriendMask(f, m) }.toArray))
                        .map(t => UserFriendsMask(t._1, t._2))
                        .flatMap(t => genPairScores(t, numPartitionsGraph, k))
                        .reduceByKey((val1, val2) => (
                            val1._1 + val2._1,
                            val1._2 + val2._2,
                            val1._3 + val2._3,
                            val1._4 + val2._4,
                            val1._5 + val2._5,
                            val1._6 + val2._6,
                            val1._7 + val2._7,
                            val1._8 + val2._8,
                            val1._9 + val2._9,
                            val1._10 | val2._10,
                            val1._11 | val2._11))
                        .map(t => PairWithScore(t._1._1, t._1._2,  // uid1, uid2
                            t._2._1, t._2._2, t._2._3, t._2._4, t._2._5, t._2._6, t._2._7, t._2._8, t._2._9, t._2._10, t._2._11))
                        .filter(pair => pair.aaScore > 1.0)
                }

                commonFriendPairs.repartition(16).toDF.write.parquet(commonFriendsPath + "/part_" + k)
                // save small part in text form for debugging
                if (k == 0) {
                    commonFriendPairs
                        .filter(pair => pair.person1 % 100 == 0)
                        .repartition(16)
                        .map(pair => (
                            pair.person1,
                            pair.person2,
                            pair.commonFriendsCount,
                            pair.aaScore,
                            pair.fedorScore,
                            pair.interactionScore,
                            pair.pageRankScore,
                            pair.commonSchool,
                            pair.commonWork,
                            pair.commonArmy,
                            pair.commonUniversity,
                            pair.maskOr,
                            pair.maskAnd
                            )
                        )
                        .saveAsTextFile(commonFriendsTextPath + "/part_" + k)
                }
            }
        }

        if (stage <= STAGE_COUNT_CITIES && !fs.exists(new Path(cityPairCountPath))) {
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

            cityPairCount.map(t => (t._1._1, t._1._2, t._2)).repartition(24).toDF().write.parquet(cityPairCountPath)
            cityPairCount.map(t => t.swap).sortByKey(ascending = false).saveAsTextFile(cityPairCountTxtPath)
        }

        val cityPairCount = {
            sqlc.read.parquet(cityPairCountPath)
                .map(t => ((t.getAs[Int](0), t.getAs[Int](1)), t.getAs[Int](2)))
        }

        val cityPairCountBC = sc.broadcast(cityPairCount.collectAsMap())

        def readPairsData(path: String) = {
            sqlc
                .read.parquet(path)
                .map(t => PairWithScore(
                    t.getAs[Int](0), t.getAs[Int](1), // uid1, uid2
                    t.getAs[Int](2), // commonFriends
                    t.getAs[Double](3), // aaScore
                    t.getAs[Double](4), // fedorScore
                    t.getAs[Double](5), // interactionScore
                    t.getAs[Int](6), // pageRankScore
                    t.getAs[Int](7), t.getAs[Int](8), t.getAs[Int](9), t.getAs[Int](10),
                    t.getAs[Int](11), t.getAs[Int](12)))
        }

//        var adamAdairPairs = sc.parallelize(Seq[PairWithScore]())
//
//        for (partNo <- fromPart until toPart + 1) {
//            // prepare data for training model
//            // step 2
//            val pairsPart = readPairsData(commonFriendsPath + "/part_" + partNo)
//            adamAdairPairs = adamAdairPairs.union(pairsPart)
//        }

        val adamAdairPairs = readPairsData(commonFriendsPath + "/part_*")
                .filter(pair => useForTraining(pair.person1) || useForTraining(pair.person2))


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

        val mainUsersFriendsBC = {
            val m = graph.map(userFriends => userFriends.user -> userFriends.friends.length)
            sc.broadcast(m.collectAsMap())
        }
        val ageSexCityBC = sc.broadcast(ageSexCity.collectAsMap())

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

        // step 5
        def prepareData(
                           adamAdairScores: RDD[PairWithScore],
                           positives: RDD[((Int, Int), Double)],
                           ageSexBC: Broadcast[scala.collection.Map[Int, AgeSexCity]],
                           cityPairCountBS: Broadcast[scala.collection.Map[(Int, Int), Int]]) = {

            adamAdairScores
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
                val friendsCount1 = mainUsersFriendsBC.value.getOrElse(pair.person1, 0)
                val friendsCount2 = mainUsersFriendsBC.value.getOrElse(pair.person2, 0)
                val jaccard = getJaccardSimilarity(pair.commonFriendsCount, friendsCount1, friendsCount2)
                val cosine = getCosineSimilarity(pair.commonFriendsCount, friendsCount1, friendsCount2)

                Vectors.dense(
                    Math.log(1.0 + pair.aaScore),
                    Math.log(1.0 + pair.fedorScore),
                    Math.log(diffAge + 1.0),
                    Math.log(meanAge),
                    jaccard,
                    cosine,
                    if (isSameSex) 1.0 else 0.0,
                    cityFactor,
                    pair.commonSchool,
                    pair.commonWork,
                    pair.commonArmy,
                    pair.commonUniversity,
                    if (pair.maskAnd > 1) 1.0 else 0.0
                )
            })
            .leftOuterJoin(positives)
        }

        val genAllPairs = false

        val allPairsFeatures = prepareData(adamAdairPairs, positives, ageSexCityBC, cityPairCountBC)

        if (genAllPairs) {
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
                features.foreachActive((i, value) => {
                    lineBuilder += value
                })
                lineBuilder.result().mkString(", ")
            }).repartition(1).saveAsTextFile(dataDir + "allPairsData_txt")
        }


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
        val testCommonFriendsCounts = readPairsData(commonFriendsPath + "/part_*")
                .filter(pair => useForPrediction(pair.person1) || useForPrediction(pair.person2))

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