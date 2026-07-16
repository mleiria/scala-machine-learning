package pt.mleiria.utils

import java.io.{File, PrintWriter}
import scala.io.Source

object IOUtils {

  private def getSource(fileName: String, encoding: String) = Source.fromFile(fileName, encoding)

  /**
   * Read a file to an Array[String]
   *
   * @param fileName
   * @param encoding
   * @return
   */
  def readFileToArray(fileName: String, encoding: String = "UTF-8"): Array[String] = {
    val source = getSource(fileName, encoding)
    val res = source.getLines().toArray
    source.close()
    res
  }

  /**
   * Read a file to a String
   *
   * @param fileName
   * @param encoding
   * @return
   */
  def readFileToString(fileName: String, encoding: String = "UTF-8"): String = {
    val source = getSource(fileName, encoding)
    val res = source.getLines().mkString
    source.close()
    res
  }

  /**
   *
   * @param fileName
   * @param encoding
   * @param splitter
   * @return
   */
  def readFileAndSplitToString(fileName: String, encoding: String = "UTF-8", splitter: String): Array[String] = {
    val source = getSource(fileName, encoding)
    val res = source.getLines().mkString.split(splitter)
    source.close()
    res
  }

  /**
   *
   * @param fileName
   * @param encoding
   * @param splitter
   * @return
   */
  def readFileToDouble(fileName: String, encoding: String = "UTF-8", splitter: String): Array[Double] = {
    val source = getSource(fileName, encoding)
    val res = source.getLines().mkString.split(splitter)
    source.close()
    res.map(_.toDouble)
  }

  /**
   *
   * @param fileName
   * @param contents
   */
  def writeToFile(fileName: String, contents: Array[String]): Unit = {
    val out = new PrintWriter(fileName)
    contents.foreach(elem => out.print(elem))
    out.close()
  }

  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  /**
   * Loads data from a file.
   *
   * @param inputPath The path to the input file.
   * @return A tuple containing the header and the data lines.
   */
  def getDataFromFile(inputPath: String): (String, Array[String]) = {
    println(s"Loading data from $inputPath...")
    val lines = readFileToArray(inputPath)
    if (lines.isEmpty) {
      println("File is empty.")
      ("", Array.empty[String])
    } else {
      (lines(0), lines.drop(1))
    }
  }

}



