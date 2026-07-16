package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import java.time.format.DateTimeFormatter

class ColumnRulesTest extends AnyFunSuite with Matchers {

  val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")

  test("cumulativeDistRule should calculate running sum of distance") {
    val rowDates = Array("2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00")
    val newHeader = Array(("distance", 0))
    val cfg = ColumnConfig(dropCol = false, rowDates = rowDates, newHeader = newHeader)
    val m = DenseMatrix(
      (10.0),
      (Double.NaN),
      (20.0)
    )

    val result = ColumnRules.cumulativeDistRule(cfg, m)
    result shouldBe DenseVector(10.0, 10.0, 30.0)
  }

  test("daysSinceStart should calculate days from first date") {
    val rowDates = Array("2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-05 00:00:00")
    val newHeader = Array(("distance", 0))
    val cfg = ColumnConfig(dropCol = false, rowDates = rowDates, newHeader = newHeader)
    val m = DenseMatrix.zeros[Double](3, 1)

    val result = ColumnRules.daysSinceStart(cfg, m)
    result shouldBe DenseVector(0.0, 1.0, 4.0)
  }

  test("distanceLast30Days should sum distances in window") {
    val rowDates = Array("2023-01-01 00:00:00", "2023-01-15 00:00:00", "2023-02-10 00:00:00")
    val newHeader = Array(("distance", 0))
    val cfg = ColumnConfig(dropCol = false, rowDates = rowDates, newHeader = newHeader)
    val m = DenseMatrix(
      (10.0),
      (20.0),
      (30.0)
    )

    val result = ColumnRules.distanceLast30Days(cfg, m)
    result shouldBe DenseVector(10.0, 30.0, 50.0)
  }
}
