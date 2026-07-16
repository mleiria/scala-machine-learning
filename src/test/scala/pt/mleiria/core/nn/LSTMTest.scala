package pt.mleiria.core.nn

import org.scalatest.funsuite.AnyFunSuite
import breeze.linalg.DenseVector


class LSTMTest extends AnyFunSuite {

  test("LSTM forward pass dimensions") {
    val inputDim = 2
    val hiddenDim = 3
    val weights = LSTMWeights.random(inputDim, hiddenDim)
    val x = Seq(DenseVector(1.0, 2.0), DenseVector(3.0, 4.0))
    val h0 = DenseVector.zeros[Double](hiddenDim)
    val c0 = DenseVector.zeros[Double](hiddenDim)

    val (hSeq, states) = LSTM.forward(x, weights, h0, c0)

    assert(hSeq.length == 2)
    assert(hSeq(0).length == hiddenDim)
    assert(states.length == 2)
  }

  test("LSTM training reduces cost on simple sequence") {
    val inputDim = 1
    val hiddenDim = 5
    val weights = LSTMWeights.random(inputDim, hiddenDim)

    // Simple task: predict the next value in a sequence
    // To match hiddenDim, we create targets of size hiddenDim where the first element is the target
    def makeTarget(v: Double): DenseVector[Double] = {
      val vec = DenseVector.zeros[Double](hiddenDim)
      vec(0) = v
      vec
    }

    val trainingData = Seq(
      (
        Seq(DenseVector(0.1), DenseVector(0.2), DenseVector(0.3)),
        Seq(makeTarget(0.2), makeTarget(0.3), makeTarget(0.4))
      ),
      (
        Seq(DenseVector(0.4), DenseVector(0.5), DenseVector(0.6)),
        Seq(makeTarget(0.5), makeTarget(0.6), makeTarget(0.7))
      )
    )

    val params = LSTMOptimizer.Hyperparameters(alpha = 0.01, iterations = 100)
    val result = LSTMOptimizer.run(trainingData, weights, params)

    val firstCost = result.costHistory(0)
    val lastCost = result.costHistory.last

    println(s"Initial cost: $firstCost, Final cost: $lastCost")
    assert(lastCost < firstCost)
  }
}
