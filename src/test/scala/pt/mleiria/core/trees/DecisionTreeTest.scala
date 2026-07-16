package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class DecisionTreeTest extends AnyFunSuite with Matchers {

  test("DecisionTree should classify simple separable data") {
    val xMatrix = DenseMatrix.tabulate(4, 2) { (r, c) =>
      if (r < 2) (if (c == 0) 1.0 else 2.0) + r * 0.1
      else (if (c == 0) 10.0 else 12.0) + r * 0.1
    }
    val y = DenseVector(0.0, 0.0, 1.0, 1.0)

    val config = TreeConfig(maxDepth = 2, minSamplesPerLeaf = 1)
    val criterion = GiniImpurity
    val tree = DecisionTree.fit(config, criterion, xMatrix, y)

    // 1.05 is between 1.0 and 1.1, so it should definitely be class 0.0
    tree.predict(DenseVector(1.05, 2.05)) shouldBe 0.0
    // 10.05 is between 10.0 and 10.1, so it should definitely be class 1.0
    tree.predict(DenseVector(10.05, 12.05)) shouldBe 1.0
  }


  test("DecisionTree should regress on simple linear data") {
    val xMatrix = DenseMatrix.tabulate(4, 1) { (r, _) => r.toDouble }
    val y = DenseVector(0.0, 1.0, 2.0, 3.0)

    val config = TreeConfig(maxDepth = 3, minSamplesPerLeaf = 1)
    val criterion = MSEImpurity
    val tree = DecisionTree.fit(config, criterion, xMatrix, y)

    tree.predict(DenseVector(0.0)) shouldBe (0.0 +- 0.5)
    tree.predict(DenseVector(3.0)) shouldBe (3.0 +- 0.5)
  }
}
