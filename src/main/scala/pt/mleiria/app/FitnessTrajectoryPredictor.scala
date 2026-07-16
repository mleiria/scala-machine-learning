package pt.mleiria.app

import breeze.linalg.{DenseMatrix, DenseVector}
import pt.mleiria.core.nn.{LSTM, LSTMOptimizer, LSTMWeights}
import pt.mleiria.utils.{Converter, IOUtils, NormalizeUtils}

import scala.collection.mutable.ArrayBuffer

object FitnessTrajectoryPredictor {

  def main(args: Array[String]): Unit = {
    val inputFilePath = "/home/manuel/Downloads/export_imputed.csv"
    val featureColumnIndex = 4 // Assuming 'distance' is at index 4 after 'start_time' is dropped

    println(s"Loading data from $inputFilePath...")
    val (header, dataLines) = IOUtils.getDataFromFile(inputFilePath)
    println(s"Data Header: $header")
    if (dataLines.isEmpty) {
      println("No data found in the file. Exiting.")
      return
    }

    // Convert data to DenseMatrix, dropping the first column (start_time)
    // Corrected: Use dropCol = true instead of colToDrop = 0
    val fullMatrix = Converter.strArrayToBreeze(dataLines, dropCol = true)

    // Extract the feature column we want to predict (e.g., distance)
    val rawFeatureData = fullMatrix(::, featureColumnIndex).toDenseVector

    // Normalize the feature data
    // getStats expects a matrix, so transpose the vector to a 1-row matrix
    val featuresX: DenseMatrix[Double] = rawFeatureData.asDenseMatrix.t
    val (mu, sigma) = NormalizeUtils.getStats(featuresX)
    val normalizedFeatureData = NormalizeUtils.applyNormalization(featuresX, mu, sigma).toDenseVector

    // Prepare data for LSTM: create sequences
    val lookBack = 10 // Number of previous time steps to use as input
    val numFeatures = 1 // We are using a single feature (distance) for now

    val trainingSequences = ArrayBuffer[(Seq[DenseVector[Double]], Seq[DenseVector[Double]])]()

    for (i <- 0 until normalizedFeatureData.length - lookBack) {
      val inputSequence: Seq[DenseVector[Double]] = (i until i + lookBack).map(j => DenseVector(normalizedFeatureData(j)))
      val targetSequence: Seq[DenseVector[Double]] = (i + 1 until i + lookBack + 1).map(j => {
        // LSTM.scala's current implementation expects target to be hiddenDim size,
        // with the actual target in the first element.
        val targetVec = DenseVector.zeros[Double](LSTMWeights.defaultHiddenDim)
        targetVec(0) = normalizedFeatureData(j)
        targetVec
      })
      trainingSequences += ((inputSequence, targetSequence))
    }

    // LSTM Model Setup
    val inputDim = numFeatures
    val hiddenDim = LSTMWeights.defaultHiddenDim // Use a default hidden dimension
    var weights = LSTMWeights.random(inputDim, hiddenDim)

    val lstmParams = LSTMOptimizer.Hyperparameters(alpha = 0.01, iterations = 500)

    println("Starting LSTM training...")
    val trainingResult = LSTMOptimizer.run(trainingSequences, weights, lstmParams)
    weights = trainingResult.weights

    println(s"LSTM training finished. Final cost: ${trainingResult.costHistory.last}")

    // --- Trajectory Projection ---
    println("\nProjecting fitness trajectory...")

    val projectionLength = 30 // Project 30 steps into the future
    val projectedTrajectory = ArrayBuffer[Double]()

    // Start projection with the last 'lookBack' values from the training data
    var currentInputSequence = (normalizedFeatureData.length - lookBack until normalizedFeatureData.length)
      .map(j => DenseVector(normalizedFeatureData(j))).toSeq

    var hPrev = DenseVector.zeros[Double](hiddenDim)
    var cPrev = DenseVector.zeros[Double](hiddenDim)

    for (_ <- 0 until projectionLength) {
      // Perform one forward pass with the current input sequence
      val (hSeq, states) = LSTM.forward(currentInputSequence, weights, hPrev, cPrev)
      val lastHiddenState = hSeq.last // Get the last hidden state

      // The prediction is assumed to be the first element of the last hidden state
      val predictedNormalizedValue = lastHiddenState(0)
      projectedTrajectory += predictedNormalizedValue

      // Update hPrev and cPrev for the next step
      hPrev = lastHiddenState
      cPrev = states.last.c // Correctly update cPrev from the last state of the forward pass

      // Create the new input sequence by dropping the oldest value and adding the new prediction
      currentInputSequence = currentInputSequence.drop(1) :+ DenseVector(predictedNormalizedValue)
    }

    // Denormalize the projected trajectory
    val denormalizedTrajectory = projectedTrajectory.map(v => v * sigma(0) + mu(0))

    println("Projected Fitness Trajectory (denormalized):")
    denormalizedTrajectory.zipWithIndex.foreach { case (value, idx) =>
      println(s"Step ${idx + 1}: ${value}")
    }

    // You can further process or visualize denormalizedTrajectory
  }
}