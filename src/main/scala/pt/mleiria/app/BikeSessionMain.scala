package pt

import org.apache.spark.sql.SparkSession
import pt.io.BikeSessionExtractor

object BikeSessionMain extends App {
  val spark = SparkSession.builder().appName("BikeDataExtraction").master("local[*]").getOrCreate()

  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"

  val bikeSessions = BikeSessionExtractor.extractBikeSessions(spark, dataDir)
  bikeSessions.show()

}
