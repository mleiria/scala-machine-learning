package pt.mleiria.core

import breeze.linalg.{DenseMatrix, DenseVector}

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
   *
   * This is a convenience wrapper that uses Linear Regression's Mean Squared Error (MSE)
   * cost function and its corresponding gradient.
   *
   * @param trainingSet The dataset containing the feature matrix (X) and target vector (y).
   * @param initW       Initial weights vector.
   * @param initB       Initial bias value.
   * @param params      Hyperparameters including learning rate (alpha) and number of iterations.
   * @return An [[OptimizationResult]] containing the optimized weights, bias, and history.
   */
  def run(trainingSet: TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: Hyperparameters): OptimizationResult = {
    run(trainingSet, initW, initB, params, LinearRegression.computeCost, LinearRegression.computeGradient)
  }

  /**
   * Performs gradient descent with customizable cost and gradient functions.
   *
   * This implementation uses a functional, tail-recursive approach to iteratively minimize
   * the cost function. In each iteration, it calculates the gradient of the cost function
   * and updates the parameters in the opposite direction of the gradient.
   *
   * @param trainingSet The dataset containing the feature matrix (X) and target vector (y).
   * @param initW       Initial weight vector.
   * @param initB       Initial bias value.
   * @param params      Optimization hyperparameters.
   * @param costFunc    A function to calculate the cost J(w, b).
   * @param gradFunc    A function to calculate the partial derivatives dJ/dw and dJ/db.
   * @return An [[OptimizationResult]] containing final parameters and the history of cost and parameters.
   */
  def run(trainingSet: TrainingSet,
          initW: DenseVector[Double],
          initB: Double,
          params: Hyperparameters,
          costFunc: CostFunc,
          gradFunc: GradFunc): OptimizationResult = {

    @scala.annotation.tailrec
    def loop(iter: Int, w: DenseVector[Double], b: Double, jHist: Vector[Double], pHist: Vector[(DenseVector[Double], Double)]): OptimizationResult = {
      if (iter >= params.iterations) {
        OptimizationResult(w, b, jHist.toArray, pHist.toArray)
      } else {
        val (dJdw, dJdb) = gradFunc(trainingSet.x, trainingSet.y, w, b)

        val nextW = w - (dJdw * params.alpha)
        val nextB = b - params.alpha * dJdb

        val currentCost = costFunc(trainingSet.x, trainingSet.y, nextW, nextB)

        if (iter % math.max(1, params.iterations / 10) == 0) {
          println(f"Iteration $iter%5d: Cost $currentCost%.4e")
        }

        loop(iter + 1, nextW, nextB, jHist :+ currentCost, pHist :+ (nextW -> nextB))
      }
    }

    loop(0, initW.copy, initB, Vector.empty, Vector.empty)
  }
}
