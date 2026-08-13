package pt.mleiria.utils

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.exp

object NormalizeUtils {

  // Function to get stats (run once on training data)
  def getStats(data: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {
    val rows = data.rows
    val rowsD = rows.toDouble

    // Column-wise mean: (data.T * ones(rows)) / rows
    val colSums = data.t * DenseVector.ones[Double](rows)
    val mu = colSums / rowsD

    // Column-wise standard deviation:
    val muMatrix = DenseMatrix.ones[Double](rows, 1) * mu.t
    val centered = data - muMatrix

    // Compute sum of squares for each column using the transpose and a sum vector
    val centeredSq = centered.map(x => x * x)
    val sumSq = centeredSq.t * DenseVector.ones[Double](rows)

    // Use Bessel's correction (n-1) for sample standard deviation
    val sigma = (sumSq / (rowsD - 1.0)).map(Math.sqrt)

    (mu, sigma)
  }

  // Function to apply existing stats (run on both training and test data)
  def applyNormalization(data: DenseMatrix[Double], mu: DenseVector[Double], sigma: DenseVector[Double]): DenseMatrix[Double] = {
    val epsilon = 1e-8
    val safeSigma = sigma + epsilon

    // Vectorized normalization using outer products to broadcast vectors to matrices
    val muMatrix = DenseMatrix.ones[Double](data.rows, 1) * mu.t
    val sigmaMatrix = DenseMatrix.ones[Double](data.rows, 1) * safeSigma.t

    (data - muMatrix) / sigmaMatrix
  }

  /**
   * Computes the sigmoid of z.
   * Supports both scalar (Double) and vector (DenseVector) inputs.
   */
  def sigmoid(z: Double): Double = {
    1.0 / (1.0 + Math.exp(-z))
  }

  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    1.0 / (exp(-z) + 1.0)
  }

  /**
   * Computes the hyperbolic tangent (tanh) of z.
   * Supports both scalar (Double) and vector (DenseVector) inputs.
   */
  def tanh(z: Double): Double = {
    Math.tanh(z)
  }

  def tanh(z: DenseVector[Double]): DenseVector[Double] = {
    z.map(Math.tanh)
  }

  /**
   * Calculates the median of a DenseVector.
   */
  def calculateMedian(v: DenseVector[Double]): Double = {
    val sorted = v.toArray.sorted
    val mid = sorted.length / 2
    if (sorted.length % 2 == 0) {
      (sorted(mid - 1) + sorted(mid)) / 2.0
    } else {
      sorted(mid)
    }
  }
}
