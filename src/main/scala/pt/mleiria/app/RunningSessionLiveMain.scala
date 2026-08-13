package pt

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, input_file_name, regexp_extract}
import pt.io.RunningSessionExtractor

object RunningSessionLiveMain extends App {
  val spark = SparkSession.builder()
    .appName("RunningSessionLiveMain")
    .master("local[*]")
    .getOrCreate()

  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"

  // 1. Extract running sessions from CSV
  val runningSessions = RunningSessionExtractor.extractRunningSessions(spark, dataDir)

  // 2. Load live data from JSON files
  // Using glob pattern to find all JSON files in the hashed directory structure
  val liveDataPath = s"$dataDir/jsons/com.samsung.shealth.exercise/*/*.json"
  val liveDataRaw = spark.read
    .option("multiLine", "true")
    .json(liveDataPath)

  // 3. Extract the UUID from the filename
  // Filename format: <uuid>.com.samsung.health.exercise.live_data.json
  val liveDataWithUuid = liveDataRaw.withColumn(
    "extracted_uuid",
    regexp_extract(input_file_name(), "([^/]+)\\.com\\.samsung\\.health\\.exercise\\.live_data\\.json$", 1)
  )

  // 4. Join Exercise sessions with their corresponding live data
  val joinedData = runningSessions.join(
    liveDataWithUuid,
    col("live_data_internal") === col("extracted_uuid"),
    "inner"
  )

  // 5. Output results
  joinedData.printSchema()
  joinedData.show()
}
