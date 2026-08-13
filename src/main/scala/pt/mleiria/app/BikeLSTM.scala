package pt.mleiria.app

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, OutputLayer}
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import pt.mleiria.utils.IOUtils.getDataFromFile
import pt.models.ActivityMetric
import pt.models.ActivityMetric.ActivityMetric

object BikeLSTM {

  def main(args: Array[String]): Unit = {

    process(ActivityMetric.MeanSpeed, "Km/h")
    process(ActivityMetric.MeanHeartRate, "bpm")
  }

  def process(targetActivityMetric: ActivityMetric, unityMeasure: String): Unit = {
    // 1. Load raw data
    val inputPath = "/home/manuel/Downloads/export_imputed.csv"
    val (header: String, data: Array[String]) = getDataFromFile(inputPath)
    println(s"Header: $header")
    println(data.mkString("Array(", "; ", ")"))

    // Indices based on the header
    val indicesNames = List(ActivityMetric.MeanHeartRate.toString, ActivityMetric.Distance.toString, ActivityMetric.Duration.toString,
      ActivityMetric.MeanSpeed.toString, ActivityMetric.Calorie.toString, ActivityMetric.AltitudeGain.toString,
      ActivityMetric.DistanceLast30Days.toString, ActivityMetric.DaysSinceLastWorkout.toString)
    val keepIndices: Array[Int] = header.split(",").zipWithIndex.filter(name => indicesNames.contains(name._1)).map(name => name._2)
    println(keepIndices.mkString("Array(", "; ", ")"))
    val targetOriginalIndex = header.split(",").zipWithIndex.filter(name => name._1 == targetActivityMetric.toString).map(name => name._2).head

    // Parse and Scale Data
    val numFeatures = keepIndices.length
    val totalRows = data.length

    val parsedFeatures = Array.ofDim[Double](totalRows, numFeatures)
    val parsedTargets = Array.ofDim[Double](totalRows)

    for (i <- 0 until totalRows) {
      val cols = data(i).split(",")
      for (f <- 0 until numFeatures) {
        parsedFeatures(i)(f) = cols(keepIndices(f)).toDouble
      }
      parsedTargets(i) = cols(targetOriginalIndex).toDouble
    }

    // Min-Max Scaling. LSTM need inputs between 0 and 1
    val fMins: Array[Double] = Array.fill(numFeatures)(Double.MaxValue)
    val fMaxs: Array[Double] = Array.fill(numFeatures)(Double.MinValue)
    var tMin = Double.MaxValue
    var tMax = Double.MinValue

    for (i <- 0 until totalRows) {
      for (f <- 0 until numFeatures) {
        if (parsedFeatures(i)(f) < fMins(f)) fMins(f) = parsedFeatures(i)(f)
        if (parsedFeatures(i)(f) > fMaxs(f)) fMaxs(f) = parsedFeatures(i)(f)
      }
      if (parsedTargets(i) < tMin) tMin = parsedTargets(i)
      if (parsedTargets(i) > tMax) tMax = parsedTargets(i)
    }

    // Apply Scaling
    val scaledFeatures = Array.ofDim[Double](totalRows, numFeatures)
    val scaledTargets = Array.ofDim[Double](totalRows)

    for (i <- 0 until totalRows) {
      for (f <- 0 until numFeatures) {
        scaledFeatures(i)(f) = (parsedFeatures(i)(f) - fMins(f)) / (fMaxs(f) - fMins(f) + 1e-8)
      }
      scaledTargets(i) = (parsedTargets(i) - tMin) / (tMax - tMin + 1e-8)
    }

    // Create the 3D Tensor(Sliding Window / Lookback)
    // We use the previous 'lookback' sessions to predict the next session
    // For this tiny dataser, we use lookback = 2
    val lookback = 2
    val numSamples = totalRows - lookback

    // ND4J Shape: [MiniBatchSize, NumFeatures, TimeSeriesLength]
    val features3D = Nd4j.zeros(numSamples, numFeatures, lookback)
    // ND4J Shape: [MiniBatchSize, OutputSize]
    val labels2D = Nd4j.zeros(numSamples, 1)

    for (i <- 0 until numSamples) {
      // Fill the timesteps for this sample
      for (t <- 0 until lookback) {
        val rowIdx = i + t
        for (f <- 0 until numFeatures) {
          features3D.putScalar(Array[Int](i, f, t), scaledFeatures(rowIdx)(f))
        }
      }
      // The target is the session AFTER the lookback window
      labels2D.putScalar(Array[Int](i, 0), scaledTargets(i + lookback))
    }

    val dataset = new DataSet(features3D, labels2D)

    // Define LSTM Neural Network Architecture
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .updater(new Adam((0.01)))
      .list()
      // Layer 0: The LSTM. It reads the sequence
      // We wrap it in 'LastTimeStep' to convert the 3D sequence output into a 2D vector for the final regression
      .layer(0, new LastTimeStep(
        new LSTM.Builder()
          .nIn(numFeatures)
          .nOut(16) // Hidden state size (memory capacity)
          .activation(Activation.TANH)
          .build()
      ))
      // Layer 1: Standard Dense Output Layer
      .layer(1, new OutputLayer.Builder(LossFunction.MSE)
        .nIn(16)
        .nOut(1)
        .activation(Activation.IDENTITY)
        .build()
      ).build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    // Print loss every 10 epochs
    model.setListeners(new ScoreIterationListener(10))

    // Train the model
    val epochs = 100
    println(" --- Starting Training --- ")
    for (_ <- 0 until epochs) {
      model.fit(dataset)
    }

    // Test and Inverse Transform
    println(s"\n --- Predictions vs Actual (Scaled back to $unityMeasure) --- ")
    val predictions = model.output(features3D)

    for(i <- 0 until numSamples){
      val actualScaled = labels2D.getDouble(i.toLong, 0.toLong)
      val predictedScaled = predictions.getDouble(i.toLong, 0.toLong)

      // De-Normalize back to real mean_speed units
      val actualResult = (actualScaled * (tMax - tMin)) + tMin
      val predictedResult = (predictedScaled * (tMax - tMin)) + tMin

      println(f"Sample $i | Actual ${targetActivityMetric.toString}: $actualResult%.2f $unityMeasure | Predicted ${targetActivityMetric.toString}: $predictedResult%.2f $unityMeasure")
    }
  }
}
