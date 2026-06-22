package pt.io

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import pt.models.SamsungHealthModels.{Movement, UserProfile}
import pt.models.SamsungHealthSchemas.{movementSchema, userProfileSchema}
import org.apache.spark.sql.functions._

object SamsungHealthDataset {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SamsungHealth")
      .master("local[*]")
      .getOrCreate()

    //val userProfileDS = readUserProfile(spark)
    //userProfileDS.show()

    val movementDS = readMovement(spark)
    movementDS.show()

    val binningData: Array[String] = movementDS.select("binning_data").collect().map(row => row.toString())
    // Get the first element
    val jsonFileName: String = binningData(0)
    println(s"Json file name: $jsonFileName")

    val fDir = "hdfs://manuel-hs:9000/user/manuel/samsung_health/jsons/com.samsung.health.movement/c/cd7c4dd4-d1ce-4c84-9593-34414856e1ab.binning_data.json"

    val df = spark.read
      .option("multiLine", "true")
      //.json("hdfs://manuel-hs:9000/user/manuel/samsung_health/jsons/com.samsung.health.movement/*/*.json")
      .json(fDir)

    val activityDf = df
      .withColumn("start_time", (col("start_time") / 1000).cast("timestamp"))
      .withColumn("end_time", (col("end_time") / 1000).cast("timestamp"))

    activityDf.show()

    spark.close()


  }

  def readMovement(spark: SparkSession): Dataset[Movement] = {
    val df = preProcessDataset(spark, movementSchema, "hdfs://manuel-hs:9000/user/manuel/samsung_health/csv/com.samsung.health.movement.20250930125884.csv")
    import spark.implicits._
    df.as[Movement]
  }

  def readUserProfile(spark: SparkSession): Dataset[UserProfile] = {
    val df = preProcessDataset(spark, userProfileSchema, "hdfs://manuel-hs:9000/user/manuel/samsung_health/csv/com.samsung.health.user_profile.20250930125884.csv")
    import spark.implicits._
    df.as[UserProfile]
  }


  private def preProcessDataset(spark: SparkSession, schema: StructType, hadoopFilePath: String): Dataset[Row] = {
    // Read the file as raw text lines
    val rawLines: Dataset[String] = spark.read.textFile(hadoopFilePath)

    val metadataLine = rawLines.first()
    println(s"Metadata line: $metadataLine")

    // Filter out the metadata line so the header is now at the top
    val cleanLines = rawLines.filter(_ != metadataLine)

    // Use the CSV reader on the Dataset[String]
    val df: Dataset[Row] = spark.read
      .option("header", "true") // Now treats the 2nd row as header
      // This tells Spark how to parse the strings into Timestamps
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSS")
      .schema(schema)
      .csv(cleanLines)
    df
  }


}
