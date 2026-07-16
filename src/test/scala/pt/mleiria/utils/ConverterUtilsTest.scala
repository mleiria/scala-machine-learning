package pt.mleiria.utils

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class ConverterUtilsTest extends AnyFunSuite with Matchers {

  test("testSConvertCorrect") {
    val initData = Array("123,456,789", "987,654,321", "0,0,0", "1,1,1")
    val matrix = Converter.strArrayToBreeze(initData)

    println(matrix)

    matrix.rows should be(4)
    matrix.cols should be(3)

  }

}
