package pt.mleiria.examples

import com.cibo.evilplot._
import com.cibo.evilplot.plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._ // Required for default styling
import com.cibo.evilplot.numeric.Point
import java.io.File

object EvilPlotExample extends App {

  val plotDir = "/home/manuel/workspace/scalaWorkspace/scala-machine-learning/plots/"

  // 1. Generate 100 random data points
  // A Point in EvilPlot takes an (X, Y) coordinate.
  val data: Seq[Point] = Seq.tabulate(100) { i =>
    Point(i.toDouble, scala.util.Random.nextDouble() * 100)
  }

  // 2. Build the plot using functional combinators
  val plot = ScatterPlot(data)
    .title("My First EvilPlot")
    .xAxis()             // Add an X-axis to the bottom
    .yAxis()             // Add a Y-axis to the left
    .frame()             // Draw a neat border box around the data
    .xGrid()             // Add vertical grid lines
    .yGrid()             // Add horizontal grid lines
    .xLabel("Index")
    .yLabel("Random Value")

  // 3. Render the plot and write it to a PNG file
  val outputFile = new File(plotDir + "scatter_plot.png")
  plot.render().write(outputFile)

  println(s"Successfully generated plot at: ${outputFile.getAbsolutePath}")
}
