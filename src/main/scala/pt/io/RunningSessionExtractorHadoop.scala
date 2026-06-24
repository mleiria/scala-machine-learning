package pt.io

import org.apache.spark.sql.{Dataset, Encoders, SparkSession}
import pt.models.SamsungHealthModels.Exercise
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}

import java.net.URI // Added Hadoop imports

/**
 * Extractor for running sessions from Samsung Health data.
 */
object RunningSessionExtractorHadoop {

  private val RUNNING_EXERCISE_TYPE = 1001

  /**
   * Extracts all running sessions from the provided Samsung Health data directory.
   *
   * @param spark The SparkSession to use for reading and filtering.
   * @param dataDir The absolute path to the Samsung Health data export folder.
   * @return A Dataset of Exercise objects filtered for running sessions.
   */
  def extractRunningSessions(spark: SparkSession, dataDir: String): Dataset[Exercise] = {
    import spark.implicits._

    // 1. Get Hadoop FileSystem instance
    val conf = spark.sparkContext.hadoopConfiguration
    //val fs = FileSystem.get(conf)
    val fs = FileSystem.get(new URI(dataDir), conf)
    val csvDir = new Path(s"$dataDir/csv")

    // 2. Find the exercise CSV file using Hadoop API
    // listStatus returns an array of FileStatus objects
    val statuses: Array[FileStatus] = fs.listStatus(csvDir)

    val exerciseFile = statuses
      .map(_.getPath) // Get the Path object for each file
      .find { p =>
        val name = p.getName // getName gets just the filename (e.g., "com.samsung...")
        name.contains("com.samsung.shealth.exercise") &&
          name.endsWith(".csv") &&
          !name.contains("custom_exercise") &&
          !name.contains("hr_zone") &&
          !name.contains("max_heart_rate") &&
          !name.contains("pacesetter") &&
          !name.contains("periodization") &&
          !name.contains("recovery_heart_rate") &&
          !name.contains("route") &&
          !name.contains("weather")
      }
      .getOrElse(throw new RuntimeException(s"Exercise CSV file not found in ${csvDir.toString}"))
      .toString // This will be the full HDFS URI

    // 3. Read the file as raw text lines (Spark handles HDFS URIs perfectly here)
    val rawLines = spark.sparkContext.textFile(exerciseFile).toDS()

    // 4. Identify and filter out the metadata line (the first line)
    val metadataLine = rawLines.first()
    val cleanLines = rawLines.filter(_ != metadataLine)

    // 5. Use the CSV reader on the Dataset[String]
    val encoder = Encoders.product[Exercise]

    val df = spark.read
      .option("header", "true")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
      .schema(encoder.schema)
      .csv(cleanLines)

    val exercises = df.as[Exercise]

    // 6. Filter for running sessions
    exercises.filter(_.exercise_type.contains(RUNNING_EXERCISE_TYPE))
  }
}