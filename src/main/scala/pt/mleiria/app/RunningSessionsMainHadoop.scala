package pt

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.desc
import pt.io.RunningSessionExtractorHadoop

object RunningSessionsMainHadoop extends App {
  val spark = SparkSession.builder().appName("RunningDataExtraction").master("local[*]").getOrCreate()

  // Hadoop file system
  val dataDirHadoop = "hdfs://manuel-hs:9000/user/manuel/samsung_health"
  val runningSessions = RunningSessionExtractorHadoop.extractRunningSessions(spark, dataDirHadoop)
  println("COUNT:" + runningSessions.count())

  //runningSessions.printSchema()
  runningSessions.orderBy(desc("start_time")).show()
}
