import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

case class AgeSexCity(age: Int, sex: Int, city: Int)

/**
  * Created by dmitry.trunin on 07.04.2016.
  */
object DemographyHelper {

    /**
      * Gets map UID -> AgeSexCity
      */
    def readAgeSexCity(sc: SparkContext,
                       dataDir: String) : RDD[(Int, AgeSexCity)] = {
        sc.textFile(Paths.getDemographyPath(dataDir))
            .map(line => {
                val lineSplit = line.trim().split("\t")
                if (lineSplit(2) == "") {
                    lineSplit(0).toInt -> AgeSexCity(0, lineSplit(3).toInt, lineSplit(5).toInt)
                }
                else {
                    lineSplit(0).toInt -> AgeSexCity(lineSplit(2).toInt, lineSplit(3).toInt, lineSplit(5).toInt)
                }
            })

    }
}
