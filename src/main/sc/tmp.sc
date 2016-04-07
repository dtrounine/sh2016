val sc: org.apache.spark.SparkContext
val sqlContext: org.apache.spark.sql.SQLContext

val data = "/mnt/ssd/all_data/interactions"

val t = sqlContext.read.parquet(data)

case class InteractionElem(index: Int, value: Double)
case class InteractionRec(from: Long, to: Long, entries: Seq[InteractionElem])

val ds = t.as[InteractionRec]

t.show(5)
val ds2 = ds.filter(_.from % 11 == 5)

ds.take(5).foreach(println)
