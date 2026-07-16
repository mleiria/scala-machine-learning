package pt.io

import org.apache.spark.sql.{Dataset, Encoders, SparkSession}
import pt.models.SamsungHealthModels.Exercise
import org.apache.spark.sql.functions.col
import pt.mleiria.utils.FilterUtils

import java.nio.file.{Files, Paths}

/**
 * Extractor for running sessions from Samsung Health data.
 */
object RunningSessionExtractor {

  private val RUNNING_EXERCISE_TYPE = 1002

  /**
   * Extracts all running sessions from the provided Samsung Health data directory.
   *
   * @param spark   The SparkSession to use for reading and filtering.
   * @param dataDir The absolute path to the Samsung Health data export folder.
   * @return A Dataset of Exercise objects filtered for running sessions.
   */
  def extractRunningSessions(spark: SparkSession, dataDir: String): Dataset[Exercise] = {
    import spark.implicits._

    // 1. Find the exercise CSV file in the csv directory
    val csvDir = s"$dataDir/csv"
    val exerciseFile = Files.list(Paths.get(csvDir))
      .filter(p =>  FilterUtils.exercisePathFilter(p))
      .findFirst()
      .orElseThrow(() => new RuntimeException(s"Exercise CSV file not found in $csvDir"))
      .toString
    println("ExerciseFile:" + exerciseFile)

    // 2. Read the file as CSV
    // We use header = false and apply the schema directly.
    // This makes the metadata line and the actual CSV header be parsed as data rows.
    // Since they won't match the Int type of exercise_type, they'll be null and filtered out.
    val encoder = Encoders.product[Exercise]
    val df = spark.read
      .option("header", "false")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
      .schema(encoder.schema)
      .csv(exerciseFile)

    val exercises = df.as[Exercise]

    // 3. Filter for running sessions using a Column expression to avoid SerializedLambda errors
    exercises.filter(col("exercise_type") === RUNNING_EXERCISE_TYPE)
  }
}
