package pt

import org.apache.spark.sql.SparkSession
import pt.io.CircuitTrainingSessionExtractor

object CircuitTrainingSessionMain extends App {
  val spark = SparkSession.builder().appName("CircuitTrainingDataExtraction").master("local[*]").getOrCreate()

  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"

  val circuitTrainingSessions = CircuitTrainingSessionExtractor.extractCircuitTrainingSessions(spark, dataDir)
  circuitTrainingSessions.printSchema()
  circuitTrainingSessions.show()

}
