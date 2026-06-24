package pt

import org.apache.spark.sql.SparkSession
import pt.models.SamsungHealthModels.Exercise
import org.apache.spark.sql.Encoders
import java.nio.file.{Files, Paths}

object VO2MaxTimelineMain extends App {
  val spark = SparkSession.builder().appName("VO2MaxTimeline").master("local[*]").getOrCreate()
  import spark.implicits._

  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"
  val csvDir = s"$dataDir/csv"

  import java.nio.file.{Files, Paths}
  val exerciseFile = Files.list(Paths.get(csvDir))
    .filter { p => p.getFileName.toString.contains("com.samsung.shealth.exercise") &&
                   p.getFileName.toString.endsWith(".csv") &&
                   !p.getFileName.toString.contains("custom_exercise") &&
                   !p.getFileName.toString.contains("hr_zone") &&
                   !p.getFileName.toString.contains("max_heart_rate") &&
                   !p.getFileName.toString.contains("pacesetter") &&
                   !p.getFileName.toString.contains("periodization") &&
                   !p.getFileName.toString.contains("recovery_heart_rate") &&
                   !p.getFileName.toString.contains("route") &&
                   !p.getFileName.toString.contains("weather")
                 }
    .findFirst()
    .orElseThrow(() => new RuntimeException("Exercise file not found"))
    .toString

  val rawLines = spark.sparkContext.textFile(exerciseFile).toDS()
  val metadataLine = rawLines.first()
  val cleanLines = rawLines.filter(_ != metadataLine)

  val encoder = Encoders.product[Exercise]
  val df = spark.read
    .option("header", "true")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
    .schema(encoder.schema)
    .csv(cleanLines)
    .as[Exercise]

  println("\n--- VO2 Max Evolution Timeline ---")
  df.filter(_.vo2_max.isDefined)
    .select("start_time", "vo2_max")
    .sort("start_time")
    .show(false)

  spark.stop()
}
