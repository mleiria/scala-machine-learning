package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.util.Random
import scala.concurrent.{Future, Await}
import scala.concurrent.ExecutionContext
import java.util.concurrent.Executors

/**
 * Random Forest implementation for Classification and Regression.
 *
 * This ensemble method builds multiple decision trees and aggregates their predictions.
 * It uses bagging (bootstrap aggregating) to ensure diversity among the trees by
 * training each tree on a random sample of the data with replacement.
 *
 * This class is immutable once created. Training is performed by the
 * companion object `RandomForest.fit`.
 *
 * @param trees The list of trained decision trees in the forest.
 * @param config The configuration parameters for the forest.
 * @param criterion The split criterion used by the trees and for final aggregation.
 */
case class RandomForest(trees: List[DecisionTree], config: RFConfig, criterion: SplitCriteria) {

  /**
   * Predicts target values for the given input matrix.
   *
   * For each sample in the matrix, the forest collects predictions from all individual
   * decision trees. These predictions are then aggregated using the split criterion:
   * - For classification: The majority vote (mode) is chosen.
   * - For regression: The average (mean) is computed.
   *
   * @param x Feature matrix where each row is a sample to be predicted.
   * @return A [[DenseVector]] containing the aggregated predictions for each sample.
   */
  def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val nSamples = x.rows

    DenseVector.tabulate(nSamples) { i =>
      val sample = x(i, ::).t
      val treePredictions = trees.map(_.predict(sample))
      criterion.aggregate(treePredictions)
    }
  }
}

/**
 * Companion object for RandomForest providing the training logic.
 */
object RandomForest {

  /**
   * Trains the Random Forest model.
   *
   * This method implements the Random Forest algorithm by training multiple decision trees
   * in parallel. To ensure diversity among the trees, it uses Bagging (Bootstrap Aggregating),
   * where each tree is trained on a random sample of the original data drawn with replacement.
   *
   * Performance Optimization: This implementation uses zero-copy bagging by passing bootstrap
   * indices to the Decision Tree training process rather than copying the feature matrix.
   *
   * @param config The configuration for the forest, including the number of trees and tree-specific settings.
   * @param x      The feature matrix.
   * @param y      The target vector.
   * @return A trained [[RandomForest]] instance.
   */
  def fit(config: RFConfig, x: DenseMatrix[Double], y: DenseVector[Double]): RandomForest = {
    val criterion = SplitCriteriaFactory.get(config.criterion)
    val nSamples = x.rows
    val nFeatures = x.cols

    // Create a thread pool for parallel tree training
    val pool = Executors.newFixedThreadPool(Runtime.getRuntime.availableProcessors())
    implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(pool)

    try {
      val treeFutures = (1 to config.numTrees).map { _ =>
        Future {
          val random = new Random()
          val bootstrapIndices = Array.fill(nSamples)(random.nextInt(nSamples))
          DecisionTree.fit(config.treeConfig, criterion, x, y, bootstrapIndices, random)
        }
      }

      val trees = Await.result(Future.sequence(treeFutures), scala.concurrent.duration.Duration.Inf).toList
      RandomForest(trees, config, criterion)
    } finally {
      pool.shutdown()
    }
  }
}
