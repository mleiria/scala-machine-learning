package pt.mleiria.core

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.pow

/**
 * Implementation of Linear Regression cost and gradient functions.
 * These can be used with the generic GradientDescent optimizer.
 */
object LinearRegression {

  /**
   * Calculates the Mean Squared Error cost for linear regression.
   */
  def computeCost(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
    val m = x.rows
    val predictions = (x * w) + b
    val errors = predictions - y
    sum(pow(errors, 2)) / (2.0 * m)
  }

  /**
   * Calculates the partial derivatives of the cost function with respect to w and b.
   */
  def computeGradient(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): (DenseVector[Double], Double) = {
    val m = x.rows.toDouble
    val errors = (x * w + b) - y
    val djdw = (x.t * errors) / m
    val djdb = sum(errors) / m
    (djdw, djdb)
  }

  /**
   * Convenience method to run Linear Regression optimization.
   */
  def run(trainingSet: GradientDescent.TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: GradientDescent.Hyperparameters): GradientDescent.OptimizationResult = {
    GradientDescent.run(trainingSet, initW, initB, params, computeCost, computeGradient)
  }
}
