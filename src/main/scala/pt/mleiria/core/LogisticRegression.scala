package pt.mleiria.core

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import pt.mleiria.utils.NormalizeUtils

/**
 * Implementation of Logistic Regression for binary classification.
 * This object provides the cost and gradient functions that can be used with
 * the generic GradientDescent optimizer.
 */
object LogisticRegression {

  /**
   * Computes the Binary Cross-Entropy cost for logistic regression.
   * J(w, b) = -1/m * sum(y * log(yHat) + (1-y) * log(1-yHat))
   */
  def computeCost(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
    val m = x.rows.toDouble
    val yHat = NormalizeUtils.sigmoid((x * w) + b)

    // Use a small epsilon to prevent log(0) = -Infinity
    val epsilon = 1e-15

    // Vectorized cost calculation
    // Cost = -1/m * sum( y * log(yHat + eps) + (1-y) * log(1 - yHat + eps) )
    val term1 = y * (yHat.map(val_ => Math.log(val_ + epsilon)))
    val term2 = (1.0 - y) * ((1.0 - yHat).map(val_ => Math.log(val_ + epsilon)))

    -sum(term1 + term2) / m
  }

  /**
   * Computes the gradient of the cost function for logistic regression.
   * djdw = 1/m * X.T * (yHat - y)
   * djdb = 1/m * sum(yHat - y)
   */
  def computeGradient(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): (DenseVector[Double], Double) = {
    val m = x.rows.toDouble
    val yHat = NormalizeUtils.sigmoid((x * w) + b)
    val error = yHat - y

    val djdw = (x.t * error) / m
    val djdb = sum(error) / m

    (djdw, djdb)
  }

  /**
   * Convenience method to run Logistic Regression optimization.
   */
  def run(trainingSet: GradientDescent.TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: GradientDescent.Hyperparameters): GradientDescent.OptimizationResult = {
    GradientDescent.run(trainingSet, initW, initB, params, computeCost, computeGradient)
  }
}
