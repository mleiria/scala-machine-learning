package pt.models
object ActivityMetric extends Enumeration {
  type ActivityMetric = Value

  val StartTime     = Value("start_time")
  val MinHeartRate  = Value("min_heart_rate")
  val MeanHeartRate = Value("mean_heart_rate")
  val MaxHeartRate  = Value("max_heart_rate")
  val TotalCalorie  = Value("total_calorie")
  val Distance      = Value("distance")
  val Duration      = Value("duration")
  val MeanSpeed     = Value("mean_speed")
  val MaxSpeed      = Value("max_speed")
  val Calorie       = Value("calorie")
  val AltitudeGain  = Value("altitude_gain")
  val CumulativeDistance = Value("cumulative_distance")
  val DaysSinceStart = Value("days_since_start")
  val DistanceLast30Days = Value("distance_last_30_days")
  val DaysSinceLastWorkout = Value("daysSinceLastWorkout")

}
