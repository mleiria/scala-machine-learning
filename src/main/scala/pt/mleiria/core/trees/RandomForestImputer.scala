package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import pt.mleiria.core.trees.ColumnRules.{cumulativeDistRule, daysSinceLastWorkout, daysSinceStart, distanceLast30Days}
import pt.mleiria.utils.Converter._
import pt.mleiria.utils.IOUtils._

import java.io.{File, PrintWriter}

/**
 * Utility to impute missing values in a CSV dataset using Random Forest regression.
 */
object RandomForestImputer {

  /**
   * Parameters for the imputation process.
   *
   * @param featureIndices Indices of columns used as features for prediction.
   * @param targetIndices  Indices of columns that need missing values imputed.
   */
  case class ImputationParams(
                               featureIndices: Seq[Int] = Seq(4, 5, 6, 7, 8),
                               targetIndices: Seq[Int] = Seq(0, 1, 2, 3, 9)
                             )

  /**
   * Configuration for the imputer.
   *
   * @param inputPath  Path to the input CSV file.
   * @param outputPath Path to save the imputed CSV file.
   * @param data       The raw data (header and lines).
   * @param datesData  The extracted date information (header: String, data: Array[String]).
   */
  case class ImputerConfig(inputPath: String, outputPath: String, data: (String, Array[String]), datesData: (String, Array[String]))

  /**
   * Main entry point for the imputation process.
   */
  def main(args: Array[String]): Unit = {
    val inputPath = "/home/manuel/Downloads/export.csv"
    val outputPath = "/home/manuel/Downloads/export_imputed.csv"
    val data = getDataFromFile(inputPath)
    val datesData = getOldestDate(data._2)
    val config = ImputerConfig(inputPath, outputPath, data, datesData)

    val resultMatrix = imputeData(config, ImputationParams())
    println(s"Imputation complete. Final matrix dimensions: ${resultMatrix.rows}x${resultMatrix.cols}")
  }

  /**
   * Imputes missing values in the matrix using Random Forest.
   *
   * The process follows these steps:
   * 1. Pre-calculates synthetic features (cumulative distance, etc.).
   * 2. For each target column identified in `ImputationParams`, trains a Random Forest
   * regressor using available data (rows without NaNs in that column).
   * 3. Uses the trained RF model to predict and fill missing values (NaNs) for that column.
   *
   * @param config The imputation configuration.
   * @param params The parameters for imputation (indices of features and targets).
   * @return A DenseMatrix containing the imputed values.
   */
  private def imputeData(config: ImputerConfig, params: ImputationParams): DenseMatrix[Double] = {
    val (header, dataLines) = config.data
    // Extract the column with dates
    val rowDates = dataLines.map(rowElem => rowElem.split(",")(0))
    // Update the header with the new field names
    val newHeader = header + ",cumulative_distance,days_since_start,distance_last_30_days, days_since_last_workout"
    // The original data in dataLines contains the timestamp in the first row. We must drop this column
    val dropFirstColumn = true
    // Create the columnConfig object
    val columnConfig = ColumnConfig(dropFirstColumn, rowDates, newHeader.split(",").zipWithIndex)
    val (_, matrix) = strArrayToBreeze(dataLines, columnConfig, Seq(cumulativeDistRule, daysSinceStart, distanceLast30Days, daysSinceLastWorkout))

    val nSamples = matrix.rows
    val nCols = matrix.cols

    // Pre-train RF models for each target column
    val models = params.targetIndices.map { tIdx =>
      val trainIndices = (0 until nSamples).filter(i => !java.lang.Double.isNaN(matrix(i, tIdx)))
      val missingIndices = (0 until nSamples).filter(i => java.lang.Double.isNaN(matrix(i, tIdx)))

      val model = if (trainIndices.nonEmpty && missingIndices.nonEmpty) {
        val (xTrain, yTrain) = extractXTrainAndYTrain(params, matrix, trainIndices, tIdx)
        val rfConfig = RFConfig(
          numTrees = 50,
          treeConfig = TreeConfig(maxDepth = 10, minSamplesPerLeaf = 2),
          criterion = "mse"
        )
        Some(RandomForest.fit(rfConfig, xTrain, yTrain))
      } else None

      tIdx -> model
    }.toMap

    val resultMatrix = DenseMatrix.tabulate(nSamples, nCols) { (i, j) =>
      val value = matrix(i, j)
      if (java.lang.Double.isNaN(value) && models.contains(j)) {
        models(j) match {
          case Some(rf) =>
            val sample = DenseVector.tabulate(params.featureIndices.length) { colIdx =>
              val fIdx = params.featureIndices(colIdx)
              val v = matrix(i, fIdx)
              if (java.lang.Double.isNaN(v)) 0.0 else v
            }
            val predMatrix = DenseMatrix.tabulate(1, params.featureIndices.length)((_, colIdx) => sample(colIdx))
            rf.predict(predMatrix)(0)
          case None => value
        }
      } else {
        value
      }
    }

    writeResultToCsv(config.outputPath, newHeader, config.datesData._2, resultMatrix)
    println(s"Final Matrix: $resultMatrix")
    resultMatrix
  }

  def extractXTrainAndYTrain(matrix: DenseMatrix[Double], trainIndices: Seq[Int], tIdx: Int)
  : (DenseMatrix[Double], DenseVector[Double])
  = {
    extractXTrainAndYTrain(ImputationParams(), matrix, trainIndices, tIdx)
  }

  def extractXTrainAndYTrain(params: ImputationParams, matrix: DenseMatrix[Double], trainIndices: Seq[Int], tIdx: Int)
  : (DenseMatrix[Double], DenseVector[Double])
  = {
    val xTrain = DenseMatrix.tabulate(trainIndices.length, params.featureIndices.length) { (r, c) =>
      val fIdx = params.featureIndices(c)
      val valOrig = matrix(trainIndices(r), fIdx)
      if (java.lang.Double.isNaN(valOrig)) 0.0 else valOrig
    }
    val yTrain = DenseVector.tabulate(trainIndices.length)(r => matrix(trainIndices(r), tIdx))
    (xTrain, yTrain)
  }

  def extractXTrainAndYTrain(params: ImputationParams, matrix: DenseMatrix[Double])
  : Map[Int, Tuple2[DenseMatrix[Double], DenseVector[Double]]] = {
    val nSamples = matrix.rows
    params.targetIndices.map { tIdx =>
      val trainIndices = (0 until nSamples).filter(i => !java.lang.Double.isNaN(matrix(i, tIdx)))
      val xTrain = DenseMatrix.tabulate(trainIndices.length, params.featureIndices.length) { (r, c) =>
        val fIdx = params.featureIndices(c)
        val valOrig = matrix(trainIndices(r), fIdx)
        if (java.lang.Double.isNaN(valOrig)) 0.0 else valOrig
      }
      val yTrain = DenseVector.tabulate(trainIndices.length)(r => matrix(trainIndices(r), tIdx))

      (tIdx, (xTrain, yTrain))
    }.toMap
  }

  /**
   * Writes the imputed matrix to a CSV file.
   *
   * @param outputPath Path to the output file.
   * @param header     The CSV header.
   * @param dates      The date column.
   * @param matrix     The matrix to write.
   */
  private def writeResultToCsv(outputPath: String, header: String, dates: Array[String], matrix: DenseMatrix[Double]): Unit = {
    val writer = new PrintWriter(new File(outputPath))
    writer.println(header)
    for (i <- 0 until matrix.rows) {
      val numericRow = (0 until matrix.cols).map { j =>
        val v = matrix(i, j)
        if (java.lang.Double.isNaN(v)) "null" else f"$v%.4f"
      }.mkString(",")
      writer.println(s"${dates(i)},$numericRow")
    }
    writer.close()
    println(s"Imputed data saved to $outputPath")
  }


  /**
   * Finds the oldest date from the data lines.
   *
   * @param dataLines The data lines of the CSV.
   * @return A tuple containing the oldest date string and the array of all date strings.
   */
  def getOldestDate(dataLines: Array[String]): (String, Array[String]) = {
    val startTimes = dataLines.map(_.split(",")(0))
    val oldestDateStr = startTimes.minBy(dateStr => convertDate(dateStr).toString)
    println(s"oldest date found: $oldestDateStr")
    (oldestDateStr, startTimes)
  }
}
