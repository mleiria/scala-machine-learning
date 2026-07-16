package pt.mleiria.core

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import breeze.linalg.{DenseMatrix, DenseVector}

class LogisticRegressionTest extends AnyFunSuite with Matchers {

  private val delta = 1e-4

  test("testComputeCostKnownValues") {
    // Simple case: m=1, x=[1], y=[1], w=[0], b=0
    // yHat = sigmoid(0) = 0.5
    // Cost = - (1 * log(0.5) + (1-1) * log(1-0.5)) = -log(0.5) = 0.693147
    val x = DenseMatrix(1.0)
    val y = DenseVector(1.0)
    val w = DenseVector(0.0)
    val b = 0.0

    val cost = LogisticRegression.computeCost(x, y, w, b)
    cost should be (0.693147 +- delta)
  }

  test("testComputeGradientKnownValues") {
    // Simple case: m=1, x=[1], y=[1], w=[0], b=0
    // yHat = sigmoid(0) = 0.5
    // error = 0.5 - 1.0 = -0.5
    // djdw = 1 * (-0.5) / 1 = -0.5
    // djdb = -0.5 / 1 = -0.5
    val x = DenseMatrix(1.0)
    val y = DenseVector(1.0)
    val w = DenseVector(0.0)
    val b = 0.0

    val (djdw, djdb) = LogisticRegression.computeGradient(x, y, w, b)
    djdw(0) should be (-0.5 +- delta)
    djdb should be (-0.5 +- delta)
  }

  test("testLogisticConvergence") {
    // Linearly separable binary data
    // Points at -2, -1 (y=0) and 1, 2 (y=1)
    val x = DenseMatrix(
      -2.0,
      -1.0,
      1.0,
      2.0
    )
    val y = DenseVector(0.0, 0.0, 1.0, 1.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.1, 5000)

    val result = LogisticRegression.run(trainingSet, initW, initB, params)

    // Cost should have decreased
    result.costHistory.last should be < result.costHistory(0)

    // The model should correctly classify the data
    // Prediction: sigmoid(x*w + b)
    val predictions = (x * result.w + result.b).map(z => 1.0 / (1.0 + Math.exp(-z)))

    predictions(0) should be < 0.5
    predictions(1) should be < 0.5
    predictions(2) should be > 0.5
    predictions(3) should be > 0.5
  }
}
