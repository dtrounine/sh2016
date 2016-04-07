import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}

/**
  * Created by dmitry.trunin on 08.04.2016.
  */
object GroupHelper {

    def readCommonGroups(sc: SparkContext,
                         dataDir: String) : RDD[((Int, Int), Double)] = {
        sc.textFile(Paths.getCommonGroupsFilename(dataDir))
            .map(line => {
                val lineSplit = line.split(",")
                val uid1 = lineSplit(0).toInt
                val uid2 = lineSplit(1).toInt
                val commonGroupScore = lineSplit(2).toDouble

                (uid1, uid2) -> commonGroupScore
            })
            .repartition(128)
    }

    def joinPairsWithGroups(sqlc: SQLContext,
                            pairs: RDD[PairWithScore],
                            groups: RDD[((Int, Int), Double)],
                            dataDir: String) : RDD[(PairWithScore, Double)] = {
        import sqlc.implicits._

        val pairGroups = pairs
            .map(pair => (pair.person1, pair.person2) -> pair)
            .fullOuterJoin(groups)
            .map(join => {
                val uids = join._1
                val pair = if (join._2._1.isDefined)
                    join._2._1.get
                else
                    PairWithScore(uids._1, uids._2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                val groupScore = join._2._2.getOrElse(0.0)

                (pair, groupScore)
            })
        pairGroups
            .toDF
            .write
            .parquet(Paths.getPairsWithGroupsPath(dataDir))

        pairGroups
    }

    def readPairWithGroups(sqlc: SQLContext,
                           dataDir: String) : RDD[(PairWithScore, Double)] = {
        sqlc
            .read
            .parquet(Paths.getPairsWithGroupsPath(dataDir))
            .map(t => {
                val d = t.getAs[Row](0)
                val groupScore = t.getAs[Double](1)

                val person1 = d.getAs[Int](0)
                val person2 = d.getAs[Int](1)
                val commonFriendsCount = d.getAs[Int](2)
                val aaScore = d.getAs[Double](3)
                val fedorScore = d.getAs[Double](4)
                val interactionScore = d.getAs[Double](5)
                val pageRankScore = d.getAs[Double](6)
                val isStrongRelation = d.getAs[Int](7)
                val isWeakRelation = d.getAs[Int](8)
                val isColleague = d.getAs[Int](9)
                val isSchoolmate = d.getAs[Int](10)
                val isArmyFellow = d.getAs[Int](11)
                val isOther = d.getAs[Int](12)
                val maskOr = d.getAs[Int](13)
                val maskAnd = d.getAs[Int](14)

                val pair = PairWithScore(
                    person1,
                    person2,
                    commonFriendsCount,
                    aaScore,
                    fedorScore,
                    interactionScore,
                    pageRankScore,
                    isStrongRelation,
                    isWeakRelation,
                    isColleague,
                    isSchoolmate,
                    isArmyFellow,
                    isOther,
                    maskOr,
                    maskAnd
                )

                (pair, groupScore)
            })
    }

}
