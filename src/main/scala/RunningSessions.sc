import org.apache.spark.sql.SparkSession
import pt.io.RunningSessionExtractor

val spark = SparkSession.builder().appName("RunningDataExtraction").master("local[*]").getOrCreate()

val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"

val runningSessions = RunningSessionExtractor.extractRunningSessions(spark, dataDir)