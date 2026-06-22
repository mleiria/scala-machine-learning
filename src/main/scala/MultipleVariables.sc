import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow

import scala.collection.mutable.ArrayBuffer

// X_train: A 3x4 Matrix
val xTrain = DenseMatrix(
  (2104.0, 5.0, 1.0, 45.0),
  (1416.0, 3.0, 2.0, 40.0),
  (852.0, 2.0, 1.0, 35.0)
)
// yTrain: A DenseVector
val yTrain = DenseVector(460.0, 232.0, 178.0)
val bInit = 785.1811367994083
val wInit = DenseVector(0.39133535, 18.75376741, -53.36032453, -26.42131618)

def predictSingleLoop(x: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
  val n = x.length
  var p = 0.0
  for (i <- 0 until n) {
    val pi = x(i) * w(i)
    p = p + pi
  }
  p = p + b
  p
}
val xVector = xTrain(0, ::).t
val fWb = predictSingleLoop(xVector, wInit, bInit)
def predict(x: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
  x.dot(w) + b
}
val fwb1 = predict(xVector, wInit, bInit)

def computeCost(x: DenseMatrix[Double], y: DenseVector[Double],
                w: DenseVector[Double], b: Double): Double = {

  val m = x.rows
  var cost = 0.0;
  for (i <- 0 until m) {
    val fWbi = x(i, ::).t.dot(w) + b
    cost = cost + pow(fWbi - y(i), 2)
  }
  cost = cost / (2 * m)
  cost
}
val cost = computeCost(xTrain, yTrain, wInit, bInit)
println(s"Cost at optimal w: $cost")

def computeGradient(x: DenseMatrix[Double], y: DenseVector[Double],
                    w: DenseVector[Double], b: Double): (DenseVector[Double], Double) = {

  // (Number of examples, number of features)
  val m = x.rows
  val n = x.cols
  var djdw = DenseVector.zeros[Double](n)
  var djdb = 0.0
  for (i <- 0 until m) {
    val err = (x(i, ::).t.dot(w) + b) - y(i)
    for (j <- 0 until n) {
      djdw(j) = djdw(j) + err * x(i, j)
    }
    djdb = djdb + err
  }
  djdw = djdw / m.toDouble
  djdb = djdb / m.toDouble
  (djdw, djdb)
}

val (tmpdjdw, tmpdjdb) = computeGradient(xTrain, yTrain, wInit, bInit)
println(s"djdb at initial w,b: $tmpdjdb")
println(s"djdw at initial w,b: $tmpdjdw.toString")

type CostFunc = (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Double) => Double
type GradFunc = (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Double) => (DenseVector[Double], Double)


def gradientDescent(x: DenseMatrix[Double], y: DenseVector[Double], wIn: DenseVector[Double],
                    bIn: Double, alpha: Double, numIters: Int,
                    costFunc: CostFunc, gradFunc: GradFunc): (DenseVector[Double], Double, Array[Double]) = {

  // An array to store cost J and w's at each iteration primarily for graphing later
  val jHistory = ArrayBuffer[Double]()
  var w = wIn
  var b = bIn

  for (i <- 0 until numIters) {
    // Calculate the gradient and update the parameters
    val (djdw, djdb) = gradFunc(x, y, w, b)
    // Update Parameters using w, b, alpha and gradient
    w = w - alpha * djdw
    b = b - alpha * djdb
    if (i < 100000) {
      jHistory += costFunc(x, y, w, b)
      if (i % math.ceil(numIters / 10) == 0) {
        println("Iteration: " + i + ": Cost: " + jHistory(jHistory.length - 1) +
          ": djdw: " + djdw + ":djdb: " + djdb + ":w: " + w + ":b: " + b)
      }
    }
  }
  (w, b, jHistory.toArray)
}

val wIn = DenseVector.zeros[Double](wInit.length)
val bIn = 0.0
val numIters = 1000
val tmpAlpha = 5.0e-7

val (w, b, jHistory) = gradientDescent(xTrain, yTrain, wIn, bIn, tmpAlpha, numIters, computeCost, computeGradient)
println(s"w, b found by gradient descent: $w, $b")


