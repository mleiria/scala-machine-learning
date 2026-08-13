package pt.mleiria

import breeze.linalg.{DenseMatrix, DenseVector}

import pt.mleiria.core.GradientDescent
import pt.mleiria.core.GradientDescent.OptimizationResult
import pt.mleiria.dto.PlotDto
import pt.mleiria.utils.{DateUtils, PlotUtils}

object LinearRegressionMain {

  def main(args: Array[String]): Unit = {
    linearReg()
    linearRegMulti()
  }


  def linearRegMulti(): Unit = {
    val xTrain = DenseMatrix((2104.0, 5.0, 1.0, 45.0), (1416.0, 3.0, 2.0, 40.0), (852.0, 2.0, 1.0, 35.0))
    val yTrain = DenseVector(460.0, 232.0, 178.0)
    val wInit = DenseVector.zeros[Double](xTrain.cols)
    val bInit = 0.0
    val numIters = 1000
    val alpha = 5.0e-7
    val hyperparameters = GradientDescent.Hyperparameters(alpha, numIters)
    val trainData = GradientDescent.TrainingSet(xTrain, yTrain)

    val result: OptimizationResult = GradientDescent.run(trainData, wInit, bInit, hyperparameters)
    println(s"Final w: ${result.w}, Final b: ${result.b}")

    val prediction = xTrain * result.w + result.b
    println(s"Prediction: $prediction, target value: $yTrain")

  }

  def linearReg(): Unit = {
    val wIn = DenseVector(0.0)
    val bIn = 0.0
    val numIters = 500
    val tmpAlpha = 1.0e-2
    val hyperparameters = GradientDescent.Hyperparameters(tmpAlpha, numIters)
    // Changed to a column matrix for a single feature with multiple samples
    val xTrain = DenseMatrix((1.0), (2.0))

    val yTrain = DenseVector(300.0, 500.0)
    val trainData = GradientDescent.TrainingSet(xTrain, yTrain)

    val result: OptimizationResult = GradientDescent.run(trainData, wIn, bIn, hyperparameters)

    // Plot Cost Vs Iteration
    val plotDto = PlotDto("lineReg_" + DateUtils.currentTimestamp, "Cost Vs. Iteration", "Iteration", "Cost")
    val iterations = (0 until numIters).toArray
    val plot = PlotUtils.linePlot(plotDto, iterations.map(_.toDouble), result.costHistory)
    PlotUtils.writeToFile(plotDto, plot)
  }

}