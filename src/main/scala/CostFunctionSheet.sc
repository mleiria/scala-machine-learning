import breeze.linalg._
import breeze.numerics._


val xTrain = DenseVector(1.0, 2.0)
val yTrain = DenseVector(300.0, 500.0)

def computeCostImperative(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): Double = {
  // Number of training examples
  val m = x.length

  var costSum = 0.0
  for (i <- 0 until m) {
    val fWb = w * x(i) + b
    val cost = math.pow(fWb - y(i), 2)
    costSum = costSum + cost
  }
  val totalCost = (1.0 / (2.0 * m)) * costSum
  totalCost
}

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

def computeCostFunc(x: DenseVector[Double], y: DenseVector[Double], w: Double, b: Double): Double = {
  val m = x.length

  // Zip x and y together, then map the cost function over the pairs
  val totalSquaredError = (x.toArray zip y.toArray).map { case (xi, yi) =>
    math.pow((w * xi + b) - yi, 2)
  }.sum

  totalSquaredError / (2.0 * m)
}

computeCost(xTrain, yTrain, 200.0, 100.0)