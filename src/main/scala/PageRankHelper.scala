import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLContext._

case class UserPageRank(user: Long, rank: Double)

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object PageRankHelper {
    def computePageRank(sc: SparkContext,
                        sqlc: SQLContext,
                        kernelUsers: Broadcast[Set[Int]],
                        dataDir: String) = {
        import sqlc.implicits._

        val numIterations = 5
        val edges = {
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
                    }.filter(f => kernelUsers.value.contains(f))
                    (user, friends)
                }).flatMap(uf => uf._2.map(x => Edge(uf._1: VertexId, x: VertexId, 1)))
        }

        Graph.fromEdges(edges, 1).staticPageRank(numIterations).vertices
            .map(v => UserPageRank(v._1, v._2))
            .toDF()
            .write.parquet(Paths.getUserPageRankPath(dataDir))
    }


}
