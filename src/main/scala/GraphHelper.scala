import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, Row}

case class UserFriends(user: Int, friends: Array[Int])

case class FriendMask(uid: Int, mask: Int)

case class UserFriendsMask(user: Int, friends: Array[FriendMask])

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object GraphHelper {

    def readGraph(sc: SparkContext, dataDir: String) = {
        sc.textFile(Paths.getGraphPath(dataDir))
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

    def readGraphWithMasks(sc: SparkContext, dataDir: String) = {
        sc.textFile(Paths.getGraphPath(dataDir))
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
            .repartition(129)
    }

    def getReversedMaskGraph(graphMask: RDD[UserFriendsMaskInteraction],
                             minEdges: Int,
                             maxEdges: Int,
                             minReversedEdges: Int,
                             maxReversedEdges: Int) = {
        graphMask
            .filter(userFriends =>
                userFriends.friends.length >= minEdges
                    && userFriends.friends.length <= maxEdges)
            .flatMap(
                userFriends => userFriends.friends.map(
                    x => (x.uid, FriendMaskInteraction(userFriends.user, x.mask, x.interaction))
                )
            )
            .groupByKey(Config.numPartitions)
            .map(t => UserFriendsMaskInteraction(t._1, t._2.toArray.sortWith((fm1, fm2) => fm1.uid < fm2.uid)))
            .filter(userFriends =>
                userFriends.friends.length >= minReversedEdges
                    && userFriends.friends.length <= maxReversedEdges)
            .map(userFriends => (userFriends.user, userFriends.friends.map(fm => (fm.uid, fm.mask, fm.interaction))))
    }

    def readReversedGraphFromParquet(sqlc: SQLContext, dataDir: String)
            : RDD[UserFriendsMaskInteraction] = {
        sqlc.read.parquet(Paths.getReversedGraphPath(dataDir))
            .map(
                (a: Row) => (
                    a.getAs[Int](0),
                    a.getAs[Seq[Row]](1).map {
                        case Row(friendUid: Int, mask: Int, interaction: Double)
                                => FriendMaskInteraction(friendUid, mask, interaction)
                    }.toArray
                )
            )
            .map(t => UserFriendsMaskInteraction(t._1, t._2))
    }
}
