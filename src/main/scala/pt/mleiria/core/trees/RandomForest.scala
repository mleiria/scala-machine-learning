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
   * For each sample, it collects predictions from all trees and aggregates them using
   * the provided split criterion (e.g., majority vote for classification, average for regression).
   *
   * @param x Feature matrix.
   * @return Vector of predictions.
   */
  def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val nSamples = x.rows

    DenseVector.tabulate(nSamples) { i =>
      val sample = DenseVector.tabulate(x.cols)(j => x(i, j))
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
   * This method trains `numTrees` decision trees in parallel using a thread pool.
   * Each tree is trained on a bootstrap sample of the original training data.
   *
   * @param config The configuration for the forest.
   * @param x Feature matrix.
   * @param y Target vector.
   * @return A trained RandomForest instance.
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
          // Bagging: Bootstrap sample of the training data
          val random = new Random()
          val bootstrapIndices = (0 until nSamples).map(_ => random.nextInt(nSamples))

          val bootX = DenseMatrix.tabulate(nSamples, nFeatures) { (i, j) =>
            x(bootstrapIndices(i), j)
          }
          val bootY = DenseVector.tabulate(nSamples) { i =>
            y(bootstrapIndices(i))
          }

          DecisionTree.fit(config.treeConfig, criterion, bootX, bootY, random)
        }
      }

      val trees = Await.result(Future.sequence(treeFutures), scala.concurrent.duration.Duration.Inf).toList
      RandomForest(trees, config, criterion)
    } finally {
      pool.shutdown()
    }
  }
}
