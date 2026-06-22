import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable.ArrayBuffer


val xTrain = DenseVector(1.0, 2.0)
val yTrain = DenseVector(300.0, 500.0)


def computeGradientImperative(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): (Double, Double) = {
  // Number of training examples
  val m = x.length
  var dJdW = 0.0
  var dJdB = 0.0
  for (i <- 0 until m) {
    val fWb = w * x(i) + b
    val dJdWi = (fWb - y(i)) * x(i)
    val dJdBi = fWb - y(i)
    dJdB += dJdBi
    dJdW += dJdWi
  }
  dJdW = dJdW / m
  dJdB = dJdB / m
  (dJdW, dJdB)
}

def computeGradient(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): (Double, Double) = {
  val m = x.length.toDouble

  // 1. Calculate all predictions: f_wb = w*x + b
  val predictions = (x * w) + b

  // 2. Calculate the errors: (f_wb - y)
  val errors = predictions - y

  // 3. dJ/dw is the dot product of errors and x, divided by m
  val djdw = (errors dot x) / m

  // 4. dJ/db is the sum of errors divided by m
  val djdb = sum(errors) / m

  (djdw, djdb)
}
def computeGradientFunc(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): (Double, Double) = {
  val m = x.length

  // Zip x and y together, calculate individual gradients, and sum them up
  val (totalDjdw, totalDjdb) = (x.toArray zip y.toArray).map { case (xi, yi) =>
    val err = (w * xi + b) - yi
    (err * xi, err) // Returns a tuple of (partial_dw, partial_db)
  }.reduce((acc, next) => (acc._1 + next._1, acc._2 + next._2))

  (totalDjdw / m, totalDjdb / m)
}

computeGradient(xTrain, yTrain, 200.0, 100)

def computeCost(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): Double = {
  val m = x.length

  // Vectorized math: f(x) = wx + b
  val predictions = (x * w) + b

  // Calculate squared errors: (predictions - y)^2
  val errors = predictions - y
  val squaredErrors = pow(errors, 2)

  // Sum and scale
  sum(squaredErrors) / (2.0 * m)
}

type CostFunc = (DenseVector[Double], DenseVector[Double], Double, Double) => Double
type GradFunc = (DenseVector[Double], DenseVector[Double], Double, Double) => (Double, Double)

def gradient_descent(x: DenseVector[Double], y: DenseVector[Double],
                     wIn: Double, bIn: Double, alpha: Double, numIters: Int,
                     costFunc: CostFunc,
                     gradFunc: GradFunc): Unit = {

  val jHistory = ArrayBuffer[Double]()
  val pHistory = ArrayBuffer[(Double, Double)]()
  var b = bIn
  var w = wIn

  for(i <- 0 until numIters){
    // Calculate the gradient and update the parameters using gradient_function
    val res = gradFunc(x, y, w, b)
    val dJdw = res._1
    val dJdb = res._2

    b = b - alpha * dJdb
    w = w - alpha * dJdw

    // Save cost J at each iteration
    if(i < 100000){
      jHistory += costFunc(x, y, w, b)
      pHistory += ((w, b))
      if(i % math.ceil(numIters/10) == 0){
        println("Iteration: " + i + ": Cost: " + jHistory(jHistory.length-1) +
          ": djdw: " + dJdw + ":djdb: " + dJdb + ":w: " + w + ":b: " + b)
      }
    }
  }
}

val wIn = 0.0
val bIn = 0.0
val numIters = 10000
val tmpAlpha = 1.0e-2

gradient_descent(xTrain, yTrain, wIn, bIn, tmpAlpha, numIters, computeCost, computeGradient)
