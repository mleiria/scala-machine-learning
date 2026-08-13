package pt.mleiria.app

import org.apache.spark.sql.{Encoders, SparkSession}
import pt.mleiria.utils.FilterUtils
import pt.models.SamsungHealthModels.Exercise


object VO2MaxTimelineMain extends App {
  val spark = SparkSession.builder().appName("VO2MaxTimeline").master("local[*]").getOrCreate()
  import spark.implicits._

  val dataDir = "/home/manuel/Downloads/Databases/samsunghealth_manuel.leiria_20250930125884"
  val csvDir = s"$dataDir/csv"

  import java.nio.file.{Files, Paths}
  val exerciseFile = Files.list(Paths.get(csvDir))
    .filter(p =>  FilterUtils.exercisePathFilter(p))
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
