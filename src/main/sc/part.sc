import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.Vector

val sc: org.apache.spark.SparkContext
val sqlContext: org.apache.spark.sql.SQLContext

val data = "/mnt/ssd/all_data/features"

val t = sqlContext.read.parquet(data)


t.map(x => (x.getAs[Row](0), x.getAs[Row](1))).map(x => (x._1.getAs[Int](0), x._1.getAs[Int](1), x._2.getAs[Double](1), x._2.getAs[Vector](0))).first()