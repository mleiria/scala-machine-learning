package pt.mleiria.utils

import java.nio.file.Path

object FilterUtils {

  private val EXCLUDED_EXERCISE_PATTERNS = Set(
    "custom_exercise",
    "hr_zone",
    "max_heart_rate",
    "pacesetter",
    "periodization",
    "recovery_heart_rate",
    "route",
    "weather",
    "challenge",
    "extension"

  )

  private def isExerciseFile(filename: String): Boolean = {
    filename.contains("com.samsung.shealth.exercise") &&
      filename.endsWith(".csv") &&
      EXCLUDED_EXERCISE_PATTERNS.forall(pattern => !filename.contains(pattern))
  }

  val exercisePathFilter: Path => Boolean = (p: Path) =>
    isExerciseFile(p.getFileName.toString)

  val exercisePathFilterHadoop: org.apache.hadoop.fs.Path => Boolean = (p: org.apache.hadoop.fs.Path) =>
    isExerciseFile(p.getName)
}
