import breeze.linalg._
import pt.mleiria.dto.PlotDto
import pt.mleiria.utils.PlotUtils

val xTrain = DenseVector(1.0, 2.0)
val yTrain = DenseVector(300.0, 500.0)

// Number of training examples
println("Number of training examples is: " + xTrain.length)

val plotDto = PlotDto("HousePrices", "Housing Prices", "Price (in 1000s of dollars)", "Size (1000 sqft)")
val scatterPlot = PlotUtils.scatterPlot(plotDto, xTrain.toArray, yTrain.toArray)
PlotUtils.writeToFile(plotDto, scatterPlot)

def computeModelOutput(x: DenseVector[Double], w: Double, b: Double): DenseVector[Double] = {
  w * x + b
}
val modelOutput: DenseVector[Double] = computeModelOutput(xTrain, 200.0, 100.0)
val plotDto = PlotDto("HousePricesPrediction", "Housing Prices Prediction", "Price (in 1000s of dollars)", "Size (1000 sqft)")
val linePlot = PlotUtils.linePlot(plotDto, xTrain.toArray, modelOutput.toArray)
PlotUtils.writeToFile(plotDto, linePlot)


val plotDto = PlotDto("multiplot", "Prediction", "x", "y");
PlotUtils.multiPlots(plotDto, scatterPlot, linePlot)
PlotUtils.writeToFile(plotDto, linePlot)