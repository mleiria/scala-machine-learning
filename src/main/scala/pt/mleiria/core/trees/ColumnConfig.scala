package pt.mleiria.core.trees

/**
 * Configuration for column-based data transformations.
 *
 * @param dropCol Whether to drop the original columns after applying transformations.
 * @param rowDates Array of date strings associated with each row, used for time-based rules.
 * @param newHeader The updated mapping of column names to their indices in the resulting matrix.
 */
case class ColumnConfig(dropCol: Boolean, rowDates: Array[String], newHeader: Array[(String, Int)]) {

}

