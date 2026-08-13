package pt.mleiria.app

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import pt.mleiria.utils.IOUtils.getDataFromFile
import pt.mleiria.utils.{Converter, NormalizeUtils}
import pt.models.ActivityMetric

object BikeAnomalyDetector {

  def main(args: Array[String]): Unit = {
    // Load raw data
    val inputPath = "/home/manuel/Downloads/export_imputed.csv"
    val (header: String, data: Array[String]) = getDataFromFile(inputPath)
    val headerData: Array[(String, Int)] = header.split(",").zipWithIndex
    println(s"Header: ${headerData.mkString("Array(", ", ", ")")}")

    val matrix = Converter.strArrayToBreeze(data, dropFirstCol = true)
    val colToRemove = headerData.find(p => p._1 == ActivityMetric.MeanHeartRate.toString).get.  _2 - 1
    val yRaw = matrix(::, colToRemove).copy
    val xRaw = matrix(::, (0 until matrix.cols).filter(_ != colToRemove)).toDenseMatrix
    val stats = NormalizeUtils.getStats(xRaw)
    val xNorm = NormalizeUtils.applyNormalization(xRaw, stats._1, stats._2)

    println("\n--- Point Anomaly Detection (Raw Values) ---")
    val pointScores = findPointAnomalies(yRaw)
    val pointThreshold = 3.0
    val pointAnomalies = pointScores.toArray.zipWithIndex.filter(_._1 > pointThreshold)

    if (pointAnomalies.isEmpty) {
      println("No point anomalies found.")
    } else {
      println(f"Found ${pointAnomalies.length} point anomalies (threshold > $pointThreshold%s):")
      pointAnomalies.foreach { case (score, index) =>
        println(f"Index $index%3d: Score $score%.4f | HeartRate ${yRaw(index)}%.2f")
      }
    }

    println("\n--- Contextual Anomaly Detection (Effort vs HR) ---")
    val contextualScores = findContextualAnomalies(xNorm, yRaw)
    val contextualThreshold = 3.0
    val contextualAnomalies = contextualScores.toArray.zipWithIndex.filter(_._1 > contextualThreshold)

    if (contextualAnomalies.isEmpty) {
      println("No contextual anomalies found.")
    } else {
      println(f"Found ${contextualAnomalies.length} contextual anomalies (threshold > $contextualThreshold%s):")
      contextualAnomalies.foreach { case (score, index) =>
        println(f"Index $index%3d: Score $score%.4f | HeartRate ${yRaw(index)}%.2f")
      }
    }
  }



  /**
   * Detects point anomalies based on the raw values of a target variable.
   *
   * It uses the Robust Z-Score (Median and MAD) to identify values that are
   * extreme relative to the rest of the dataset, regardless of context.
   *
   * @param y The target vector (e.g., Heart Rate).
   * @return A vector of Robust Z-scores.
   */
  def findPointAnomalies(y: DenseVector[Double]): DenseVector[Double] = {
    val median = NormalizeUtils.calculateMedian(y)
    val absDeviations = abs(y - median)
    val mad = NormalizeUtils.calculateMedian(absDeviations)

    val consistencyConstant = 0.6745
    val safeMad = mad + 1e-8

    (consistencyConstant * (y - median)) / safeMad
  }

  // Assuming X_norm is your NORMALIZED Breeze matrix (Speed, Altitude, Distance, etc.)
  // and y_raw is your TARGET vector (Heart Rate)
  def findContextualAnomalies(X_norm: DenseMatrix[Double], y_raw: DenseVector[Double]): DenseVector[Double] = {

    // --- Phase 1: Robust Weight Estimation (Trimmed Least Squares) ---
    // 1. Initial fit to get a baseline
    val wInitial = X_norm \ y_raw
    val residualsInitial = abs(y_raw - (X_norm * wInitial))

    // 2. Identify a "clean" subset (ignore top 5% most extreme residuals)
    val sortedResiduals = residualsInitial.toArray.sorted
    val thresholdResidual = sortedResiduals((sortedResiduals.length * 0.95).toInt)

    val inlierIndices = residualsInitial.toArray.zipWithIndex.filter(_._1 < thresholdResidual).map(_._2).toArray

    // 3. Refit the model using only the inliers
    // We create sub-matrices for the inliers
    val xInliers = DenseMatrix.tabulate(inlierIndices.length, X_norm.cols) { (i, j) =>
      X_norm(inlierIndices(i), j)
    }
    val yInliers = DenseVector.tabulate(inlierIndices.length) { i =>
      y_raw(inlierIndices(i))
    }

    val wRobust = xInliers \ yInliers

    // --- Phase 2: Robust Anomaly Scoring ---
    // 4. Predict expected Heart Rate using the robust model
    val y_pred = X_norm * wRobust
    val residuals = abs(y_raw - y_pred)

    // 5. Calculate Robust Z-Score using Median Absolute Deviation (MAD)
    val medianResidual = NormalizeUtils.calculateMedian(residuals)
    val absDeviations = abs(residuals - medianResidual)
    val mad = NormalizeUtils.calculateMedian(absDeviations)

    val consistencyConstant = 0.6745
    val safeMad = mad + 1e-8

    (consistencyConstant * (residuals - medianResidual)) / safeMad
  }

}
