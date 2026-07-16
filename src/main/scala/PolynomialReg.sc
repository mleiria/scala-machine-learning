import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow
import pt.mleiria.core.GradientDescent
import pt.mleiria.core.GradientDescent.OptimizationResult

// y = 1 + X^2
val X = DenseMatrix.tabulate(20, 1)((row, col) => row.toDouble)
// y should be a DenseVector for linear regression
val y = DenseVector.tabulate(20)(row => 1.0 + row.toDouble * row.toDouble)

val trainingSet = GradientDescent.TrainingSet(X, y)
val initW = DenseVector.zeros[Double](1)
val initB = 0.0
val params = GradientDescent.Hyperparameters(1e-5, 10000)

val result: OptimizationResult = GradientDescent.run(trainingSet, initW, initB, params)
val yHat = X * result.w + result.b

case class Result(x: Double, y: Double, yHat: Double)

val rows = (0 until X.rows).map(i =>
  Result(x = X(i, 0), y = y(i), yHat = yHat(i))
)
pow(X,2)