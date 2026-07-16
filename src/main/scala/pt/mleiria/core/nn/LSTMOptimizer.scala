package pt.mleiria.core.nn

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Optimizer for LSTM networks using Stochastic Gradient Descent.
 */
object LSTMOptimizer {

  case class Hyperparameters(alpha: Double, iterations: Int)

  case class OptimizationResult(
    weights: LSTMWeights,
    costHistory: Array[Double]
  )

  /**
   * Trains the LSTM model on a set of sequences.
   *
   * @param trainingData A sequence of (inputSequence, targetSequence) pairs.
   * @param initWeights Initialized LSTM weights.
   * @param params       Optimization hyperparameters.
   * @return OptimizationResult containing the trained weights and cost history.
   */
  def run(trainingData: Seq[(Seq[DenseVector[Double]], Seq[DenseVector[Double]])],
          initWeights: LSTMWeights,
          params: Hyperparameters): OptimizationResult = {

    var weights = initWeights
    val costHistory = new scala.collection.mutable.ArrayBuffer[Double]()

    val hiddenDim = weights.Wf.rows

    for (iter <- 0 until params.iterations) {
      val totalGradWf = DenseMatrix.zeros[Double](weights.Wf.rows, weights.Wf.cols)
      val totalGradBf = DenseVector.zeros[Double](weights.bf.length)
      val totalGradWi = DenseMatrix.zeros[Double](weights.Wi.rows, weights.Wi.cols)
      val totalGradBi = DenseVector.zeros[Double](weights.bi.length)
      val totalGradWc = DenseMatrix.zeros[Double](weights.Wc.rows, weights.Wc.cols)
      val totalGradBc = DenseVector.zeros[Double](weights.bc.length)
      val totalGradWo = DenseMatrix.zeros[Double](weights.Wo.rows, weights.Wo.cols)
      val totalGradBo = DenseVector.zeros[Double](weights.bo.length)

      var iterationCost = 0.0

      for ((x, target) <- trainingData) {
        // Forward pass
        val (hSeq, states) = LSTM.forward(x, weights, DenseVector.zeros[Double](hiddenDim), DenseVector.zeros[Double](hiddenDim))

        // Cost
        iterationCost += LSTM.computeCost(hSeq, target)

        // Backward pass (BPTT)
        val grads = LSTM.backward(x, target, states, weights)

        // Accumulate gradients
        totalGradWf += grads.Wf
        totalGradBf += grads.bf
        totalGradWi += grads.Wi
        totalGradBi += grads.bi
        totalGradWc += grads.Wc
        totalGradBc += grads.bc
        totalGradWo += grads.Wo
        totalGradBo += grads.bo
      }

      val avgCost = iterationCost / trainingData.length
      costHistory += avgCost

      val n = trainingData.length.toDouble
      // Update weights (SGD)
      weights = LSTMWeights(
        Wf = weights.Wf - (totalGradWf / n) * params.alpha,
        bf = weights.bf - (totalGradBf / n) * params.alpha,
        Wi = weights.Wi - (totalGradWi / n) * params.alpha,
        bi = weights.bi - (totalGradBi / n) * params.alpha,
        Wc = weights.Wc - (totalGradWc / n) * params.alpha,
        bc = weights.bc - (totalGradBc / n) * params.alpha,
        Wo = weights.Wo - (totalGradWo / n) * params.alpha,
        bo = weights.bo - (totalGradBo / n) * params.alpha
      )

      if (iter % math.max(1, params.iterations / 10) == 0) {
        println(f"Iteration $iter%5d: Average Cost $avgCost%.4e")
      }
    }

    OptimizationResult(weights, costHistory.toArray)
  }
}
