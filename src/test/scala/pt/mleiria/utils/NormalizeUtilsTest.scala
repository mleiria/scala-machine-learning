package pt.mleiria.utils

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import breeze.linalg.{DenseMatrix, DenseVector}

class NormalizeUtilsTest extends AnyFunSuite with Matchers {

  private val delta = 1e-6

  test("testStatsCorrectness") {
    val data = DenseMatrix.tabulate(2, 2) { (r, c) =>
      if (r == 0) {
        if (c == 0) 10.0 else 20.0
      } else {
        if (c == 0) 20.0 else 40.0
      }
    }

    val (mu, sigma) = NormalizeUtils.getStats(data)

    // Col 0: [10, 20] -> mean = 15, stddev = sqrt(((10-15)^2 + (20-15)^2)/1) = sqrt(50) = 7.0710678
    // Col 1: [20, 40] -> mean = 30, stddev = sqrt(((20-30)^2 + (40-30)^2)/1) = sqrt(200) = 14.1421356

    mu(0) should be (15.0 +- delta)
    mu(1) should be (30.0 +- delta)
    sigma(0) should be (7.0710678 +- delta)
    sigma(1) should be (14.1421356 +- delta)
  }

  test("testBasicNormalization") {
    val data = DenseMatrix.tabulate(2, 2) { (r, c) =>
      if (r == 0) {
        if (c == 0) 1.0 else 2.0
      } else {
        if (c == 0) 3.0 else 4.0
      }
    }

    val (mu, sigma) = NormalizeUtils.getStats(data)
    val normalized = NormalizeUtils.applyNormalization(data, mu, sigma)

    // For each column, the normalized mean should be 0 and stddev 1
    for (c <- 0 until normalized.cols) {
      val col = normalized(::, c)
      val mean = breeze.stats.mean(DenseVector(col.toArray))
      val stddev = breeze.stats.stddev(DenseVector(col.toArray))

      mean should be (0.0 +- delta)
      stddev should be (1.0 +- delta)
    }
  }

  test("testZeroVariance") {
    // Column 2 is constant (5.0)
    val data = DenseMatrix.tabulate(2, 2) { (r, c) =>
      if (c == 0) {
        if (r == 0) 1.0 else 3.0
      } else {
        5.0
      }
    }

    val (mu, sigma) = NormalizeUtils.getStats(data)
    val normalized = NormalizeUtils.applyNormalization(data, mu, sigma)

    // Constant column should not result in NaN
    for (r <- 0 until normalized.rows) {
      normalized(r, 1).isNaN should be (false)
      normalized(r, 1).isInfinity should be (false)
      // (5.0 - 5.0) / (0 + epsilon) = 0
      normalized(r, 1) should be (0.0 +- delta)
    }
  }

  test("testSigmoidScalar") {
    NormalizeUtils.sigmoid(0.0) should be (0.5 +- delta)
    NormalizeUtils.sigmoid(100.0) should be (1.0 +- delta)
    NormalizeUtils.sigmoid(-100.0) should be (0.0 +- delta)
  }

  test("testSigmoidVector") {
    val z = DenseVector(0.0, 2.0, -2.0)
    val result = NormalizeUtils.sigmoid(z)

    result(0) should be (0.5 +- delta)
    result(1) should be (0.880797 +- delta)
    result(2) should be (0.119203 +- delta)
  }
}
