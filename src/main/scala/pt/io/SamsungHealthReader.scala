package pt.io

import org.apache.spark.sql.{Dataset, SparkSession, DataFrame}
import org.apache.spark.sql.functions.{col, trim, from_csv}
import org.apache.spark.sql.Encoders
import pt.models.SamsungHealthModels._

object SamsungHealthReader {

  private val BASE_PATH = "hdfs://manuel-hs:9000/user/manuel/samsung_health/csv"

  /**
    * Reads a Samsung Health CSV file from HDFS and returns it as a Dataset[T].
    * Handles the specific format where the first line is metadata and the second line is the header.
    */
  def readDataset[T](spark: SparkSession, fileName: String)(implicit encoder: org.apache.spark.sql.Encoder[T]): Dataset[T] = {
    val path = s"$BASE_PATH/$fileName"

    // 1. Read as text to avoid the CSV reader truncating columns based on the metadata row
    val textDf = spark.read.text(path)

    // 2. Collect the first two lines to identify metadata and header
    val firstTwoLines = textDf.limit(2).collect().map(_.getString(0))
    if (firstTwoLines.length < 2) {
      throw new RuntimeException(s"File $fileName does not contain enough rows (metadata + header).")
    }

    val metaLine = firstTwoLines(0).trim()
    val headerLine = firstTwoLines(1).trim()

    // 3. Filter out the metadata and header rows
    val dataLinesDf = textDf.filter(trim(col("value")) =!= metaLine && trim(col("value")) =!= headerLine)

    // 4. Parse the CSV strings using the schema from the encoder
    // from_csv requires a Map of options (like separator)
    val options = Map("sep" -> ",")
    val parsedDf = dataLinesDf.select(
      from_csv(col("value"), encoder.schema, options)
    ).select("col.*") // Flatten the resulting struct

    parsedDf.as[T]
  }

  // --- Helper methods for specific models ---

  def readUserProfile(spark: SparkSession, fileName: String): Dataset[UserProfile] =
    readDataset[UserProfile](spark, fileName)(Encoders.product[UserProfile])

  def readWeight(spark: SparkSession, fileName: String): Dataset[Weight] =
    readDataset[Weight](spark, fileName)(Encoders.product[Weight])

  def readExercise(spark: SparkSession, fileName: String): Dataset[Exercise] =
    readDataset[Exercise](spark, fileName)(Encoders.product[Exercise])

  def readSleep(spark: SparkSession, fileName: String): Dataset[Sleep] =
    readDataset[Sleep](spark, fileName)(Encoders.product[Sleep])

  def readStepCount(spark: SparkSession, fileName: String): Dataset[TrackerPedometerStepCount] =
    readDataset[TrackerPedometerStepCount](spark, fileName)(Encoders.product[TrackerPedometerStepCount])

  def readHeartRate(spark: SparkSession, fileName: String): Dataset[TrackerHeartRate] =
    readDataset[TrackerHeartRate](spark, fileName)(Encoders.product[TrackerHeartRate])

  def readVitalityScore(spark: SparkSession, fileName: String): Dataset[VitalityScore] =
    readDataset[VitalityScore](spark, fileName)(Encoders.product[VitalityScore])

  def readActivityDaySummary(spark: SparkSession, fileName: String): Dataset[ActivityDaySummary] =
    readDataset[ActivityDaySummary](spark, fileName)(Encoders.product[ActivityDaySummary])
}
