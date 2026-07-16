package pt.io

import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Encoders, SparkSession}
import pt.mleiria.utils.FilterUtils
import pt.models.SamsungHealthModels.Exercise

import java.net.URI

/**
 * Extractor for running sessions from Samsung Health data.
 */
object RunningSessionExtractorHadoop {

  private val RUNNING_EXERCISE_TYPE = 1002

  /**
   * Extracts all running sessions from the provided Samsung Health data directory.
   *
   * @param spark   The SparkSession to use for reading and filtering.
   * @param dataDir The absolute path to the Samsung Health data export folder.
   * @return A Dataset[Exercise] filtered for running sessions.
   */
  def extractRunningSessions(spark: SparkSession, dataDir: String): Dataset[Exercise] = {
    import spark.implicits._

    // 1. Get Hadoop FileSystem instance
    val conf = spark.sparkContext.hadoopConfiguration
    val fs = FileSystem.get(new URI(dataDir), conf)
    val csvDir = new Path(s"$dataDir/csv")

    // 2. Find the exercise CSV file using Hadoop API
    val statuses: Array[FileStatus] = fs.listStatus(csvDir)

    val exerciseFile = statuses
      .map(_.getPath)
      .find { p =>
        FilterUtils.exercisePathFilterHadoop(p)
      }
      .getOrElse(throw new RuntimeException(s"Exercise CSV file not found in ${csvDir.toString}"))
      .toString // This will be the full HDFS URI

    println("ExerciseFile:" + exerciseFile)

    // 3. Read the file as CSV
    // We use header = false and apply the the schema directly.
    // This makes the metadata line and the actual CSV header be parsed as data rows.
    // Since they won't match the Int type of exercise_type, they'll be null and filtered out.
    val encoder = Encoders.product[Exercise]
    val df = spark.read
      .option("header", "false")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
      .schema(encoder.schema)
      .csv(exerciseFile)

    val exercises = df.as[Exercise]

    // 4. Filter for running sessions using a Column expression to avoid SerializedLambda errors
    exercises.filter(col("exercise_type") === RUNNING_EXERCISE_TYPE)
  }
}
