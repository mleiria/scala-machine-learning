package pt.mleiria.utils

import com.cibo.evilplot._
import com.cibo.evilplot.colors.HTMLNamedColors
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.plot.{ScatterPlot, _}
import pt.mleiria.dto.PlotDto

import java.io.File

object PlotUtils {
  val plotDir = "/home/manuel/workspace/scalaWorkspace/scala-machine-learning/plots/"


  def scatterPlot(plotDto: PlotDto, x: Array[Double], y: Array[Double]): Plot = {
    val data = fromDoubleToSeq(x, y)
    // Build the plot using functional combinators
    val plot = ScatterPlot(data)
      .title(plotDto.title)
      .xAxis() // Add an X-axis to the bottom
      .yAxis() // Add a Y-axis to the left
      .frame() // Draw a neat border box around the data
      .xGrid() // Add vertical grid lines
      .yGrid() // Add horizontal grid lines
      .xLabel(plotDto.xLabel)
      .yLabel(plotDto.yLabel)
      .background(HTMLNamedColors.white)
    plot
    //writeToFile(plotDto, plot)
  }

  def linePlot(plotDto: PlotDto, x: Array[Double], y: Array[Double]): Plot = {
    val data = fromDoubleToSeq(x, y)
    // Build the plot using functional combinators
    val plot = LinePlot.series(data, plotDto.title, HTMLNamedColors.blue)
      .title(plotDto.title)
      .xAxis() // Add an X-axis to the bottom
      .yAxis() // Add a Y-axis to the left
      .frame() // Draw a neat border box around the data
      .xGrid() // Add vertical grid lines
      .yGrid() // Add horizontal grid lines
      .xLabel(plotDto.xLabel)
      .yLabel(plotDto.yLabel)
      .background(HTMLNamedColors.white)
    plot
    //writeToFile(plotDto, plot)
  }

  def multiPlots(plotDto: PlotDto, plots: Plot*): Plot = {
    // Overlay them
    val combinedPlot = Overlay(plots: _*)
      .xAxis() // Add X axis
      .yAxis() // Add Y axis
      .frame() // Add a border frame
      .xLabel(plotDto.xLabel)
      .yLabel(plotDto.yLabel)
      .title(plotDto.title)
    combinedPlot
  }

  private def fromDoubleToSeq(x: Array[Double], y: Array[Double]): Seq[Point] =
    Seq.tabulate(x.length) { i => Point(x(i), y(i)) }

  def writeToFile(plotDto: PlotDto, plot: Plot): Unit = {
    //  Render the plot and write it to a PNG file
    val outputFile = new File(plotDir + plotDto.fileName + ".png")
    plot.render().write(outputFile)

    println(s"Successfully generated plot at: ${outputFile.getAbsolutePath}")
  }
}
