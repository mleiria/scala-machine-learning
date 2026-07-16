package pt.mleiria.core.trees

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class SplitCriteriaTest extends AnyFunSuite with Matchers {

  test("GiniImpurity should calculate correctly") {
    // Pure set
    GiniImpurity.computeImpurity(Seq(1.0, 1.0, 1.0)) shouldBe 0.0
    // Balanced set: 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5
    GiniImpurity.computeImpurity(Seq(1.0, 0.0)) shouldBe 0.5
    // Mixed set: 1 - ( (2/3)^2 + (1/3)^2 ) = 1 - (4/9 + 1/9) = 4/9 ≈ 0.4444
    GiniImpurity.computeImpurity(Seq(1.0, 1.0, 0.0)) shouldBe (4.0/9.0 +- 1e-6)
  }

  test("EntropyImpurity should calculate correctly") {
    // Pure set
    EntropyImpurity.computeImpurity(Seq(1.0, 1.0, 1.0)) shouldBe 0.0
    // Balanced set: - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (-0.5 - 0.5) = 1.0
    EntropyImpurity.computeImpurity(Seq(1.0, 0.0)) shouldBe 1.0
  }

  test("MSEImpurity should calculate correctly") {
    // Constant set
    MSEImpurity.computeImpurity(Seq(10.0, 10.0, 10.0)) shouldBe 0.0
    // Simple mixed set: mean=5, MSE = ( (2-5)^2 + (8-5)^2 ) / 2 = (9 + 9)/2 = 9.0
    MSEImpurity.computeImpurity(Seq(2.0, 8.0)) shouldBe 9.0 +- 1e-6
  }

  test("Classification leaf value should be mode") {
    GiniImpurity.calculateLeafValue(Seq(1.0, 1.0, 0.0)) shouldBe 1.0
    EntropyImpurity.calculateLeafValue(Seq(0.0, 0.0, 1.0)) shouldBe 0.0
  }

  test("Regression leaf value should be mean") {
    MSEImpurity.calculateLeafValue(Seq(2.0, 8.0)) shouldBe 5.0
  }

  test("Aggregation should work as expected") {
    // Classification: mode
    GiniImpurity.aggregate(Seq(1.0, 0.0, 1.0)) shouldBe 1.0
    // Regression: mean
    MSEImpurity.aggregate(Seq(10.0, 20.0)) shouldBe 15.0
  }
}
