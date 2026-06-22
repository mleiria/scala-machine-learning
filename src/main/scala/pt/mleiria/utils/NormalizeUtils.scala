package pt.mleiria.utils

import breeze.linalg.DenseMatrix

object NormalizeUtils {

  def zScoreNormalization(data: DenseMatrix[Double]): DenseMatrix[Double] = {
    val m = data.rows
    val n = data.cols
    val normalizedData = DenseMatrix.zeros[Double](m, n)

    for (j <- 0 until n) {
      val col = data(::, j)
      val mu = breeze.stats.mean(col)
      val sigma = breeze.stats.stddev(col)

      for (i <- 0 until m) {
        normalizedData(i, j) = (data(i, j) - mu) / sigma
      }
    }
    normalizedData
  }

}
