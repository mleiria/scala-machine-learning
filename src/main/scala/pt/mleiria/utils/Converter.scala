package pt.mleiria.utils

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import pt.mleiria.core.trees.ColumnConfig

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit


object Converter extends Serializable {

  case class Milliseconds(value: Int) extends AnyVal

  case class Seconds(value: Double) extends AnyVal

  case class Minutes(value: Double) extends AnyVal

  case class Hours(value: Double) extends AnyVal

  case class Meters(value: Double) extends AnyVal

  case class Kilometers(value: Double) extends AnyVal

  def convertToSeconds(ms: Milliseconds): Seconds = Seconds(ms.value / 1000.0)

  def convertToMinutes(ms: Milliseconds): Minutes = Minutes(ms.value / 60000.0)

  def convertToHours(ms: Milliseconds): Hours = Hours(ms.value / 3600000.0)

  def convertToKm(m: Meters): Kilometers = Kilometers(m.value / 1000.0)

  /**
   * Convert a date string in the format yyyy-MM-dd HH:mm:ss.SSS to LocalDateTime
   *
   * @param dateStr date string in the format yyyy-MM-dd HH:mm:ss.SSS
   * @return LocalDateTime object corresponding to the input parameter
   */
  def convertDate(dateStr: String): LocalDateTime = {
    // Define the pattern matching your string format
    // 'S' represents fractional seconds (up to 3 digits for milliseconds)
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")

    // Handle cases where fractional seconds might be shorter than 3 digits (e.g., .0 or .55)
    val normalizedDateStr = if (dateStr.contains(".")) {
      val parts = dateStr.split("\\.")
      val datePart = parts(0)
      val fractionPart = if (parts.length > 1) parts(1) else ""
      val paddedFraction = fractionPart.padTo(3, '0')
      s"$datePart.$paddedFraction"
    } else {
      s"$dateStr.000"
    }

    // Parse to LocalDateTime
    val localDateTime = LocalDateTime.parse(normalizedDateStr, formatter)

    localDateTime
  }

  def findDaysBetweenDates(startDateTime: LocalDateTime, endDateTime: LocalDateTime): Long = {

    // Convert to LocalDate (drops the time part)
    val startDate = startDateTime.toLocalDate
    val endDate = endDateTime.toLocalDate

    // Calculate the difference
    val daysBetween = ChronoUnit.DAYS.between(startDate, endDate)

    //println(s"Days between: $daysBetween")
    // Output: Days between: 5
    daysBetween
  }


  def findDaysBetweenDates(startStr: String, endStr: String): Long = {
    // Reuse the robust convertDate method to handle potential formatting inconsistencies
    val startDateTime = convertDate(startStr)
    val endDateTime = convertDate(endStr)

    // Convert to LocalDate (drops the time part)
    val startDate = startDateTime.toLocalDate
    val endDate = endDateTime.toLocalDate

    // Calculate the difference
    val daysBetween = ChronoUnit.DAYS.between(startDate, endDate)
    // Output: Days between: 5
    daysBetween
  }


  // 80-96, 97-112, 113-128, 129-144, 145-160
  val findZone = (x: Double) => x match {
    case v if v >= 145 && v <= 200 => 5
    case v if v >= 129 && v < 145 => 4
    case v if v >= 113 && v < 129 => 3
    case v if v >= 97 && v < 113 => 2
    case v if v >= 80 && v < 97 => 1
    case _ => 0
  }

  /**
   * Transforms a Breeze DenseMatrix to a Spark DataFrame
   */
  def breezeToSpark(spark: SparkSession, matrix: DenseMatrix[Double], headers: Array[String]): org.apache.spark.sql.DataFrame = {

    // 1. Create the Schema using the headers
    // Every column in your Breeze matrix is a Double
    val schema = StructType(
      headers.map(header => StructField(header, DoubleType, nullable = true))
    )

    // 2. Convert Breeze rows to a local Seq of Spark Rows
    // We iterate through the rows of the matrix
    val rows = (0 until matrix.rows).map { i =>
      // matrix(i, ::) gets the i-th row as a Breeze Vector
      // .toArray converts it to a standard Scala Array
      Row.fromSeq(matrix(i, ::).inner.toArray)
    }

    // 3. Parallelize and create the DataFrame
    val rdd = spark.sparkContext.parallelize(rows)
    spark.createDataFrame(rdd, schema)
  }

  /**
   * Functional parser that allows injecting custom feature engineering
   *
   * @param extraFeatures A sequence of lambdas: (BaseMatrix) => NewColumnVector
   */
  def strArrayToBreeze(dataLines: Array[String],
                       columnConfig: ColumnConfig,
                       extraFeatures: Seq[(ColumnConfig, DenseMatrix[Double]) => DenseVector[Double]] = Seq.empty
                      ): (ColumnConfig, DenseMatrix[Double]) = {

    // 1. Base Parsing
    val parsedData = dataLines.map { line =>
      splitter(line, columnConfig.dropCol).map(convertToDouble)
    }
    val newColumnConfig = if (columnConfig.dropCol)
      ColumnConfig(columnConfig.dropCol, columnConfig.rowDates, columnConfig.newHeader.drop(1))
    else columnConfig

    val nSamples = parsedData.length
    if (nSamples == 0) return (newColumnConfig, DenseMatrix.zeros[Double](0, 0))
    val nCols = parsedData(0).length

    // Create the base matrix
    var matrix = DenseMatrix.tabulate(nSamples, nCols) { (i, j) => parsedData(i)(j) }

    // 2. Functional Feature Engineering
    // For each lambda, we calculate a new column and horizontally concatenate it
    extraFeatures.foreach { fn =>
      val newColumn = fn(newColumnConfig, matrix)
      // Reshape vector to (nSamples x 1) and concatenate
      matrix = DenseMatrix.horzcat(matrix, newColumn.asDenseMatrix.reshape(nSamples, 1))
    }

    (newColumnConfig, matrix)
  }

  def strArrayToBreeze(dataLines: Array[String], dropCol: Boolean): DenseMatrix[Double] = {
    val parsedData = dataLines.map { line =>
      splitter(line, dropCol = dropCol).map(cell => convertToDouble(cell))
    }
    val nSamples = parsedData.length
    if (nSamples == 0) return DenseMatrix.zeros[Double](0, 0)
    val nCols = parsedData(0).length
    DenseMatrix.tabulate(nSamples, nCols) { (i, j) => parsedData(i)(j) }
  }

  def strArrayToBreeze(dataLines: Array[String]): DenseMatrix[Double] = {
    strArrayToBreeze(dataLines, dropCol = false)
  }


  def splitter(line: String, dropCol: Boolean): Array[String] = {
    if (dropCol) line.split(",").drop(1) else line.split(",")
  }

  private def convertToDouble(cell: String): Double = {
    if (cell == null || cell.trim == "null" || cell.trim == "") {
      Double.NaN
    } else {
      try {
        cell.trim.toDouble
      } catch {
        case _: Exception => Double.NaN
      }
    }
  }

}