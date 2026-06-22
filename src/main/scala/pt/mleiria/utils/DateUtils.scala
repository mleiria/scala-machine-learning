package pt.mleiria.utils

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object DateUtils {
  private val formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmm")

  def currentTimestamp: String = LocalDateTime.now().format(formatter)
}