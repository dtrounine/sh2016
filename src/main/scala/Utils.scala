import org.apache.spark.broadcast.Broadcast

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object Utils {
    def useForTraining(uid: Int): Boolean = {
        uid % 11 == 5
    }

    def useForPrediction(uid: Int): Boolean = {
        uid % 11 == 7
    }

    def useUser(uid: Int): Boolean = {
        useForTraining(uid) || useForPrediction(uid)
    }

    def filterPair(pair: PairWithScore,
                   neighborCountBC: Broadcast[scala.collection.Map[Int, Int]]) : Boolean = {
        val friends1 = neighborCountBC.value.getOrElse(pair.person1, 0)
        val friends2 = neighborCountBC.value.getOrElse(pair.person2, 0)

        friends1 >= 10 && friends2 >= 10 && pair.aaScore > 0.5 ||
        (friends1 < 10 || friends2 < 10) && pair.aaScore > 0.25
    }
}
