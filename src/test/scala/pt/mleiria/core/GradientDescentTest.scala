package pt.mleiria.core

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import breeze.linalg.{DenseMatrix, DenseVector}

class GradientDescentTest extends AnyFunSuite with Matchers {

  private val delta = 1e-2

  test("testSimpleLinearConvergence") {
    val data = Array(
      1.0, 2.0,
      2.0, 1.0,
      3.0, 3.0,
      4.0, 1.0
    )
    val x = new DenseMatrix[Double](4, 2)
    for (r <- 0 until 4; c <- 0 until 2) {
      x(r, c) = data(r * 2 + c)
    }
    val y = DenseVector(13.0, 12.0, 20.0, 16.0)

    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](2)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 10000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (2.0 +- delta)
    result.w(1) should be (3.0 +- delta)
    result.b should be (5.0 +- delta)
  }

  test("testCostDecreases") {
    val data = Array(1.0, 2.0, 3.0)
    val x = new DenseMatrix[Double](3, 1)
    for (r <- 0 until 3) {
      x(r, 0) = data(r)
    }
    val y = DenseVector(2.0, 4.0, 6.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 100)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.costHistory.last should be < result.costHistory(0)
  }

  test("testCustomCostAndGradient") {
    val x = new DenseMatrix[Double](1, 1)
    x(0, 0) = 1.0
    val y = DenseVector(1.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.1, 10)

    val mockCost = (_: DenseMatrix[Double], _: DenseVector[Double], _: DenseVector[Double], _: Double) => 0.0
    val mockGrad = (_: DenseMatrix[Double], _: DenseVector[Double], _: DenseVector[Double], _: Double) => (DenseVector(1.0), 1.0)

    val result = GradientDescent.run(trainingSet, initW, initB, params, mockCost, mockGrad)

    result.w(0) should be (-1.0 +- delta)
    result.b should be (-1.0 +- delta)
    result.costHistory.forall(_ == 0.0) should be (true)
  }


  test("testExactLinear1D") {
    val x = DenseMatrix.tabulate(3, 1)((r, _) => (r + 1).toDouble)
    val y = DenseVector(3.0, 5.0, 7.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 2000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (2.0 +- delta)
    result.b should be (1.0 +- delta)
  }

  test("testNegativeLinear1D") {
    val x = DenseMatrix.tabulate(3, 1)((r, _) => (r + 1).toDouble)
    val y = DenseVector(-7.0, -9.0, -11.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 10000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (-2.0 +- delta)
    result.b should be (-5.0 +- delta)
  }

  test("testHighDimConvergence") {
    val x = DenseMatrix.tabulate(4, 3) { (r, c) =>
      if (r == 0) 1.0
      else if (r == c + 1) 1.0
      else 0.0
    }
    val y = DenseVector(10.0, 5.0, 6.0, 7.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](3)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 5000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (1.0 +- delta)
    result.w(1) should be (2.0 +- delta)
    result.w(2) should be (3.0 +- delta)
    result.b should be (4.0 +- delta)
  }

  test("testZeroFeatures") {
    val x = DenseMatrix.zeros[Double](2, 1)
    val y = DenseVector(5.0, 5.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.1, 1000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (0.0 +- delta)
    result.b should be (5.0 +- delta)
  }

  test("testNonZeroInit") {
    val x = DenseMatrix.tabulate(3, 1)((r, _) => (r + 1).toDouble)
    val y = DenseVector(3.0, 5.0, 7.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector(10.0)
    val initB = 10.0
    val params = GradientDescent.Hyperparameters(0.01, 10000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (2.0 +- delta)
    result.b should be (1.0 +- delta)
  }

  test("testMinimumDataset") {
    val x = DenseMatrix.tabulate(2, 1)((r, _) => (r + 1).toDouble)
    val y = DenseVector(3.0, 5.0)
    val trainingSet = GradientDescent.TrainingSet(x, y)
    val initW = DenseVector.zeros[Double](1)
    val initB = 0.0
    val params = GradientDescent.Hyperparameters(0.01, 10000)

    val result = GradientDescent.run(trainingSet, initW, initB, params)

    result.w(0) should be (2.0 +- delta)
    result.b should be (1.0 +- delta)
  }
}
