package pt.io

import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Encoders
import pt.models.SamsungHealthModels.Exercise
import java.nio.file.{Files, Paths}

/**
 * Extractor for bike sessions from Samsung Health data.
 */
object BikeSessionExtractor {

  private val BIKE_EXERCISE_TYPES = Set(1002, 11007)

  /**
    * Extracts all bike sessions from the provided Samsung Health data directory.
    *
    * @param spark The SparkSession to use for reading and filtering.
    * @param dataDir The absolute path to the Samsung Health data export folder.
    * @return A Dataset of Exercise objects filtered for bike sessions.
    */
  def extractBikeSessions(spark: SparkSession, dataDir: String): Dataset[Exercise] = {
    import spark.implicits._

    // 1. Find the exercise CSV file in the csv directory
    val csvDir = s"$dataDir/csv"
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
      .orElseThrow(() => new RuntimeException(s"Exercise CSV file not found in $csvDir"))
      .toString

    // 2. Read the file as raw text lines to handle the metadata row
    val rawLines = spark.sparkContext.textFile(exerciseFile).toDS()

    // 3. Identify and filter out the metadata line (the first line)
    val metadataLine = rawLines.first()
    val cleanLines = rawLines.filter(_ != metadataLine)

    // 4. Use the CSV reader on the Dataset[String]
    val encoder = Encoders.product[Exercise]

    val df = spark.read
      .option("header", "true")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
      .schema(encoder.schema)
      .csv(cleanLines)

    val exercises = df.as[Exercise]

    // 5. Filter for bike sessions (1002: Native, 11007: Generic/Imported)
    exercises.filter(e => e.exercise_type.exists(t => BIKE_EXERCISE_TYPES.contains(t)))
  }
}
