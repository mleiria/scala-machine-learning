package pt

import org.apache.spark.sql.SparkSession
import pt.io.RunningSessionExtractorHadoop


object RunningSessionsMain extends App {
  val spark = SparkSession.builder().appName("RunningDataExtraction").master("local[*]").getOrCreate()

  // Local file system
  //val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"
  //val runningSessions = RunningSessionExtractor.extractRunningSessions(spark, dataDirHadoop)

  // Hadoop file system
  val dataDirHadoop = "hdfs://manuel-hs:9000/user/manuel/samsung_health"
  val runningSessions = RunningSessionExtractorHadoop.extractRunningSessions(spark, dataDirHadoop)


  runningSessions.printSchema()
  runningSessions.show()
}
