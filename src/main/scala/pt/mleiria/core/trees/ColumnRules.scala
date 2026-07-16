package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import pt.mleiria.utils.Converter

/**
 * A set of rules for calculating synthetic features from raw data matrices.
 * Rules are defined as functions that take a [[ColumnConfig]] and a [[DenseMatrix]]
 * and return a [[DenseVector]] of the calculated values.
 */
object ColumnRules {

  /**
   * Calculates the cumulative distance over time.
   *
   * @return A vector where each element is the sum of distances up to that row.
   */
  val cumulativeDistRule: (ColumnConfig, DenseMatrix[Double]) => DenseVector[Double] = (cfg: ColumnConfig, m: DenseMatrix[Double]) => {
    val distCol = m(::, cfg.newHeader.filter(name => name._1.equals("distance"))(0)._2)
    val sums = distCol.toArray.scanLeft(0.0)((acc, x) => acc + (if (x.isNaN) 0.0 else x)).tail
    DenseVector(sums)
  }

  /**
   * Calculates the number of days since the start of the recording.
   *
   * @return A vector where each element is the days difference between the first row's date and the current row.
   */
  val daysSinceStart: (ColumnConfig, DenseMatrix[Double]) => DenseVector[Double] = (cfg: ColumnConfig, m: DenseMatrix[Double]) => {
    val oldestDate = cfg.rowDates(0)
    DenseVector.tabulate(m.rows)(i => Converter.findDaysBetweenDates(oldestDate, cfg.rowDates(i)))
  }

  val daysSinceLastWorkout: (ColumnConfig, DenseMatrix[Double]) => DenseVector[Double] = (cfg: ColumnConfig, m: DenseMatrix[Double]) => {
    DenseVector.tabulate(m.rows)(i => if( i == 0) 0 else Converter.findDaysBetweenDates(cfg.rowDates(i - 1), cfg.rowDates(i)))
  }

  /**
   * Calculates the sum of distances in the last 30 days relative to each row.
   *
   * @return A vector where each element is the total distance covered in the 30-day window preceding that row.
   */
  val distanceLast30Days: (ColumnConfig, DenseMatrix[Double]) => DenseVector[Double] = (cfg: ColumnConfig, m: DenseMatrix[Double]) => {
    val distCol = m(::, cfg.newHeader.filter(name => name._1.equals("distance"))(0)._2)
    val rowDates = cfg.rowDates
    val localDates = rowDates.map(Converter.convertDate)

    DenseVector.tabulate(m.rows) { i =>
      val currentDate = localDates(i)
      val thirtyDaysAgo = currentDate.minusDays(30)

      (i to 0 by -1)
        .takeWhile(j => !localDates(j).isBefore(thirtyDaysAgo))
        .map(j => if (distCol(j).isNaN) 0.0 else distCol(j))
        .sum
    }
  }

  /**
   * Rule to create a bias (intercept) column of ones.
   */
  val biasRule = (columnConfig: ColumnConfig, m: DenseMatrix[Double]) => DenseVector.ones[Double](m.rows)

}