import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}

case class InteractionElem(index: Int, value: Double)
case class InteractionRec(from: Long, to: Long, entries: Seq[InteractionElem])

case class FriendMaskInteraction(uid: Int, mask: Int, interaction: Double)

case class UserFriendsMaskInteraction(user: Int, friends: Array[FriendMaskInteraction])

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object InteractionsHelper {
    def getInteractionScore(interactionEntries: Seq[(Int, Double)]): Double = {
        var sum = 0.0
        for (interaction <- interactionEntries) {
            val iType = interaction._1
            val importance =
                if (iType == 1) 0.0 // удаление фида из ленты
                else if (iType == 2) 1.0 // поход в гости
                else if (iType == 3) 1.0 // участие в опросе
                else if (iType == 4) 8.0 // отправка личного сообщения
                else if (iType == 5) 0.0 // удаление личного сообщения
                else if (iType == 6) 2.0 // класс объекта
                else if (iType == 7) 0.0 // разкласс объекта
                else if (iType == 8) 4.0 // комментирование пользовательского поста
                else if (iType == 9) 4.0 // комментирования пользовательского фото
                else if (iType == 10) 4.0 // комментирование пользовательского видео
                else if (iType == 11) 4.0 // комментирование фотоальбома
                else if (iType == 12) 1.0 // класс к комментарию
                else if (iType == 13) 1.0 // отправка сообщения на форуме
                else if (iType == 14) 8.0 // оценка фото
                else if (iType == 15) 1.0 // просмотр фото
                else if (iType == 16) 16.0 // отметка пользователя на фотографиях
                else if (iType == 17) 16.0 // отметка пользователя на отдельном фото
                else if (iType == 18) 32.0 // отправка подарка
                else 0.0
            sum += Math.log(1.0 + importance * interaction._2)
        }
        sum
    }

    def readInteractions(sqlc: SQLContext, dataDir: String) : RDD[((Int, Int), Double)]= {
       sqlc.read.parquet(Paths.getInteractionsPath(dataDir))
               .map((row: Row) => (row.getAs[Long](0), row.getAs[Long](1), row.getAs[Seq[Row]](2).map{
                   case Row(index: Int, value: Double) => (index, value)
               }))
           .map(x => (x._1.toInt, x._2.toInt) -> getInteractionScore(x._3))
    }

    def joinGraphAndInteraction(graph: RDD[UserFriendsMask],
                                interactions: RDD[((Int, Int), Double)]) : RDD[UserFriendsMaskInteraction] = {
        graph.flatMap(
                userFriends => userFriends.friends.map(
                    x => ((userFriends.user, x.uid), x.mask)
                )
            )
            .leftOuterJoin(interactions)
            .map(t => {
                val key = t._1
                val userUid = key._1
                val friendUid = key._2
                val mask = t._2._1
                val interaction = t._2._2.getOrElse(0.0)

                userUid -> FriendMaskInteraction(friendUid, mask, interaction)
            })
            .groupByKey(Config.numPartitions)
            .map(t => UserFriendsMaskInteraction(t._1, t._2.toArray))
    }

    def readGraphInteractionFromParquet(sqlc: SQLContext, dataDir: String) : RDD[UserFriendsMaskInteraction] = {
        sqlc.read
            .parquet(Paths.getGraphInteractionPath(dataDir))
            .map((a: Row) => UserFriendsMaskInteraction(
                user = a.getAs[Int](0),
                friends = a.getAs[Seq[Row]](1).map {
                    case Row(friendUid: Int, mask: Int, interaction: Double)
                        => FriendMaskInteraction(friendUid, mask, interaction)
                }.toArray)
            )
    }
}
