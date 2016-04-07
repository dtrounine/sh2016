import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object CityHelper {

    def calcCityPairCounts(sc: SparkContext,
                           sqlc: SQLContext,
                           dataDir: String,
                           graph: RDD[UserFriends]) : Unit = {
        import sqlc.implicits._

        val userCity = {
            sc.textFile(Paths.getDemographyPath(dataDir))
                .map(line => {
                    val lineSplit = line.trim().split("\t")
                    lineSplit(0).toInt -> lineSplit(5).toInt
                })
        }

        val cityPopulation =
            userCity
                .map(t => t._2 -> 1)
                .reduceByKey((x, y) => x + y)
        cityPopulation
            .map(t => t.swap)
            .sortByKey(ascending = false)
            .saveAsTextFile(Paths.getCityPopulationTxtPath(dataDir))

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

        cityPairCount.map(t => (t._1._1, t._1._2, t._2)).repartition(24).toDF().write.parquet(Paths.getCityPairCountPath(dataDir))
        cityPairCount.map(t => t.swap).sortByKey(ascending = false).saveAsTextFile(Paths.getCityPairCountTxtPath(dataDir))
    }
}
