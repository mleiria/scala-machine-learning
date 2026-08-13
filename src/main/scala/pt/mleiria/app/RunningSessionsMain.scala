package pt.mleiria.app

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.desc
import pt.io.RunningSessionExtractor

object RunningSessionsMain extends App {
  val spark = SparkSession.builder().appName("RunningDataExtraction").master("local[*]").getOrCreate()

  // Local file system
  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20260625153006"
  val runningSessions = RunningSessionExtractor.extractRunningSessions(spark, dataDir)
  println("COUNT:" + runningSessions.count())

  //runningSessions.printSchema()
  runningSessions.orderBy(desc("start_time")).show()
}
