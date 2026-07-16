package pt.mleiria.core.trees


/**
 * Trait for calculating the impurity or loss of a split and aggregating values.
 */
trait SplitCriteria {
  /**
   * Indicates whether this criterion is used for regression.
   */
  def isRegression: Boolean

  /**
   * Calculates the impurity of a set of labels.
   *
   * @param labels The labels (target values) in the current node.
   * @return The impurity value.
   */
  def computeImpurity(labels: Seq[Double]): Double

  /**
   * Calculates the representative value for a leaf node.
   *
   * @param labels The labels in the current node.
   * @return The representative value (e.g., mode for classification, mean for regression).
   */
  def calculateLeafValue(labels: Seq[Double]): Double

  /**
   * Aggregates predictions from multiple trees in a Random Forest.
   *
   * @param predictions The predictions from individual trees.
   * @return The aggregated prediction (e.g., mode for classification, mean for regression).
   */
  def aggregate(predictions: Seq[Double]): Double
}

/**
 * Base trait for classification criteria, providing default implementations
 * for leaf value calculation and aggregation using the mode.
 */
trait ClassificationCriteria extends SplitCriteria {
  override def isRegression: Boolean = false

  override def calculateLeafValue(labels: Seq[Double]): Double = {
    if (labels.isEmpty) 0.0
    else labels.groupBy(identity).maxBy(_._2.size)._1
  }

  override def aggregate(predictions: Seq[Double]): Double = {
    if (predictions.isEmpty) 0.0
    else predictions.groupBy(identity).maxBy(_._2.size)._1
  }
}

/**
 * Gini Impurity for classification.
 * G = 1 - sum(pi^2)
 */
object GiniImpurity extends ClassificationCriteria {
  override def computeImpurity(labels: Seq[Double]): Double = {
    if (labels.isEmpty) 0.0
    else {
      val n = labels.length.toDouble
      val counts = labels.groupBy(identity).mapValues(_.size.toDouble)
      1.0 - counts.values.map(p => Math.pow(p / n, 2)).sum
    }
  }
}

/**
 * Information Entropy for classification.
 * H = -sum(pi * log2(pi))
 */
object EntropyImpurity extends ClassificationCriteria {
  override def computeImpurity(labels: Seq[Double]): Double = {
    if (labels.isEmpty) 0.0
    else {
      val n = labels.length.toDouble
      val counts = labels.groupBy(identity).mapValues(_.size.toDouble)
      -counts.values.map(p => (p / n) * Math.log(p / n) / Math.log(2)).sum
    }
  }
}

/**
 * Mean Squared Error for regression.
 * MSE = 1/n * sum((yi - y_mean)^2)
 */
object MSEImpurity extends SplitCriteria {
  override def isRegression: Boolean = true

  override def computeImpurity(labels: Seq[Double]): Double = {
    if (labels.isEmpty) 0.0
    else {
      val n = labels.length.toDouble
      val mean = labels.sum / n
      labels.map(y => Math.pow(y - mean, 2)).sum / n
    }
  }

  override def calculateLeafValue(labels: Seq[Double]): Double = {
    if (labels.isEmpty) 0.0
    else labels.sum / labels.length
  }

  override def aggregate(predictions: Seq[Double]): Double = {
    if (predictions.isEmpty) 0.0
    else predictions.sum / predictions.length
  }
}

/**
 * Factory for creating split criteria based on a string identifier.
 */
object SplitCriteriaFactory {
  /**
   * Returns the SplitCriteria implementation corresponding to the given identifier.
   *
   * @param criterion The identifier ("gini", "entropy", "mse").
   * @return An implementation of SplitCriteria.
   * @throws IllegalArgumentException if the criterion is unknown.
   */
  def get(criterion: String): SplitCriteria = criterion.toLowerCase match {
    case "gini"    => GiniImpurity
    case "entropy" => EntropyImpurity
    case "mse"     => MSEImpurity
    case _         => throw new IllegalArgumentException(s"Unknown criterion: $criterion")
  }
}
