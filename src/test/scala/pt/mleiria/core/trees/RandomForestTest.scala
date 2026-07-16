package pt.mleiria.core.trees

import org.scalatest.funsuite.AnyFunSuite
import breeze.linalg.{DenseMatrix, DenseVector, sum}


class RandomForestTest extends AnyFunSuite {

  test("RandomForest regression reduces error on synthetic data") {
    // Synthetic linear-ish data: y = 2x + 1 + noise
    val nSamples = 100
    val x = DenseMatrix.zeros[Double](nSamples, 1)
    val y = DenseVector.zeros[Double](nSamples)

    for (i <- 0 until nSamples) {
      val valX = i.toDouble / 10.0
      x(i, 0) = valX
      y(i) = 2.0 * valX + 1.0 + (scala.util.Random.nextDouble() * 0.1)
    }

    val config = RFConfig(
      numTrees = 20,
      treeConfig = TreeConfig(maxDepth = 5, minSamplesPerLeaf = 2),
      criterion = "mse"
    )
    val rf = RandomForest.fit(config, x, y)

    val predictions = rf.predict(x)

    // Compute MSE
    val diff = predictions - y
    val mse = sum(diff * diff) / nSamples

    println(s"Training MSE: $mse")
    assert(mse < 1.0) // Should be reasonably low for this simple data
  }

  test("RandomForest classification works on simple separable data") {
    // Separable data: x < 0.5 -> 0, x >= 0.5 -> 1
    val nSamples = 50
    val x = DenseMatrix.zeros[Double](nSamples, 1)
    val y = DenseVector.zeros[Double](nSamples)

    for (i <- 0 until nSamples) {
      val valX = scala.util.Random.nextDouble()
      x(i, 0) = valX
      y(i) = if (valX < 0.5) 0.0 else 1.0
    }

    val config = RFConfig(
      numTrees = 10,
      treeConfig = TreeConfig(maxDepth = 3),
      criterion = "gini"
    )
    val rf = RandomForest.fit(config, x, y)

    val predictions = rf.predict(x)
    var correct = 0
    for (i <- 0 until nSamples) {
      if (predictions(i) == y(i)) correct += 1
    }

    val accuracy = correct.toDouble / nSamples
    println(s"Classification Accuracy: $accuracy")
    assert(accuracy > 0.8)
  }
}
