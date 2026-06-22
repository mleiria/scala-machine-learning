package pt.mleiria.core

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.pow

import scala.collection.mutable.ArrayBuffer

/**
 * Implementation of the Gradient Descent algorithm for Linear Regression.
 *
 * Gradient descent is an optimization algorithm used to minimize a cost function by iteratively
 * moving in the direction of steepest descent.
 */
object GradientDescent {

  /**
   * Data container for training features and targets.
   * @param x Matrix of training features (m x n).
   * @param y Vector of target values (m).
   */
  case class TrainingSet(x: DenseMatrix[Double], y: DenseVector[Double])

  /**
   * Encapsulates hyperparameters for the optimization process.
   * @param alpha Learning rate.
   * @param iterations Number of iterations to perform.
   */
  case class Hyperparameters(alpha: Double, iterations: Int)

  /**
   * Encapsulates the results of the gradient descent optimization.
   * @param w Final weight vector.
   * @param b Final bias value.
   * @param costHistory Array of cost values at each iteration.
   * @param parameterHistory Array of (w, b) pairs at each iteration.
   */
  case class OptimizationResult(
    w: DenseVector[Double],
    b: Double,
    costHistory: Array[Double],
    parameterHistory: Array[(DenseVector[Double], Double)]
  )

  type CostFunc = (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Double) => Double
  type GradFunc = (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Double) => (DenseVector[Double], Double)

  /**
   * Performs gradient descent using default cost and gradient implementations.
   */
  def run(trainingSet: TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: Hyperparameters): OptimizationResult = {
    run(trainingSet, initW, initB, params, computeCost, computeGradient)
  }

  /**
   * Performs gradient descent with customizable cost and gradient functions.
   *
   * @param trainingSet The dataset containing x and y vectors.
   * @param initW       Initial weight vector.
   * @param initB       Initial bias value.
   * @param params      Optimization hyperparameters.
   * @param costFunc    Function to calculate the cost J(w, b).
   * @param gradFunc    Function to calculate the partial derivatives dJ/dw and dJ/db.
   * @return OptimizationResult containing final parameters and history.
   */
  def run(trainingSet: TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: Hyperparameters,
          costFunc: CostFunc,
          gradFunc: GradFunc): OptimizationResult = {

    val jHistory = new ArrayBuffer[Double](params.iterations)
    val pHistory = new ArrayBuffer[(DenseVector[Double], Double)](params.iterations)
    var b = initB
    var w = initW.copy

    for (i <- 0 until params.iterations) {
      val (dJdw, dJdb) = gradFunc(trainingSet.x, trainingSet.y, w, b)

      // Simultaneous update of w and b
      w = w - (dJdw * params.alpha)
      b = b - params.alpha * dJdb

      // Record history
      val currentCost = costFunc(trainingSet.x, trainingSet.y, w, b)
      jHistory += currentCost
      pHistory += (w.copy -> b)

      if (i % math.max(1, params.iterations / 10) == 0) {
        println(f"Iteration $i%5d: Cost $currentCost%.4e")
      }
    }
    OptimizationResult(w, b, jHistory.toArray, pHistory.toArray)
  }

  /**
   * Calculates the Mean Squared Error cost for linear regression.
   */
  private def computeCost(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
    val m = x.rows
    val predictions = (x * w) + b
    val errors = predictions - y
    sum(pow(errors, 2)) / (2.0 * m)
  }

  /**
   * Calculates the partial derivatives of the cost function with respect to w and b.
   */
  private def computeGradient(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): (DenseVector[Double], Double) = {
    val m = x.rows.toDouble
    val errors = (x * w + b) - y
    val djdw = (x.t * errors) / m
    val djdb = sum(errors) / m
    (djdw, djdb)
  }
}
