package pt.mleiria.examples

import org.apache.spark.sql.SparkSession
/**
 * The command line to run spark is:
 *
 * spark-submit --class "pt.mleiria.examples.SimpleApp" --master "local[*]" target/scala-2.12/scala-machine-learning_2.12-0.1.0-SNAPSHOT.jar
 */
object SimpleApp {
  def main(args: Array[String]): Unit = {
    // 1. Initialize the SparkSession
    val spark = SparkSession.builder
      .appName("Simple Analysis App")
      .master("local[*]") // Use all local CPU cores
      .getOrCreate()

    import spark.implicits._

    // 2. Create some sample data (Dataframe)
    val data = Seq(
      ("Scala", 100),
      ("Python", 90),
      ("Java", 80),
      ("Scala", 150)
    ).toDF("Language", "Score")

    // 3. Perform a simple transformation (Group by and Sum)
    val result = data.groupBy("Language")
      .sum("Score")
      .withColumnRenamed("sum(Score)", "TotalScore")

    // 4. Show the result
    println("--- Aggregated Data ---")
    result.show()

    // 5. Stop the session
    spark.stop()
  }
}