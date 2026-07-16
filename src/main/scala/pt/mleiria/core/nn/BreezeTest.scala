package pt.mleiria.core.nn

import breeze.linalg.{DenseMatrix, DenseVector}

object BreezeTest {
  def main(args: Array[String]): Unit = {
    val v1 = DenseVector(1.0, 2.0)
    val v2 = DenseVector(3.0, 4.0)
    val m = DenseMatrix((1.0, 2.0), (3.0, 4.0))

    // Let's test operators
    val eMul = v1 * v2
    // val eMul2 = v1 :* v2
    val mm = m * v1

    println("v1 * v2 type: " + eMul.getClass.getName)
    println("m * v1 type: " + mm.getClass.getName)
  }
}
