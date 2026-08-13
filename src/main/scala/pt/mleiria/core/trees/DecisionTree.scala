package pt.mleiria.core.trees

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.util.Random

/**
 * Node structure for a Decision Tree.
 */
sealed trait Node
case class InternalNode(featureIdx: Int, threshold: Double, left: Node, right: Node) extends Node
case class LeafNode(value: Double) extends Node

/**
 * Represents a potential split in the decision tree.
 *
 * @param gain The impurity reduction achieved by this split.
 * @param featureIdx The index of the feature used for splitting.
 * @param threshold The value used to split the feature.
 * @param leftIndices Indices of samples that go to the left child.
 * @param rightIndices Indices of samples that go to the right child.
 */
case class Split(gain: Double, featureIdx: Int, threshold: Double, leftIndices: Array[Int], rightIndices: Array[Int])

/**
 * Implementation of a Decision Tree for Classification and Regression.
 *
 * This class is immutable once created. Training is performed by the
 * companion object `DecisionTree.fit`.
 *
 * @param root The root node of the decision tree.
 * @param config The configuration used during training.
 * @param criterion The split criterion used to train the tree.
 */
case class DecisionTree(root: Node, config: TreeConfig, criterion: SplitCriteria) {

  /**
   * Predicts the target value for a given input sample by traversing the tree
   * from the root to a leaf node.
   *
   * @param x The feature vector for which to predict.
   * @return The predicted target value (mode for classification, mean for regression).
   */
  def predict(x: DenseVector[Double]): Double = {
    def traverse(node: Node): Double = node match {
      case LeafNode(v) => v
      case InternalNode(fIdx, threshold, left, right) =>
        if (x(fIdx) <= threshold) traverse(left) else traverse(right)
    }
    traverse(root)
  }
}

/**
 * Companion object for DecisionTree providing the training logic.
 */
object DecisionTree {

  /**
   * Trains a Decision Tree model using the full provided feature matrix.
   *
   * @param config   The configuration parameters for the tree (max depth, min samples per leaf, etc.).
   * @param criterion The impurity measure to use for splitting (e.g., Gini, Entropy, MSE).
   * @param x        The feature matrix where each row is a sample.
   * @param y        The target vector.
   * @param random   Random instance for feature sampling, used for Random Forest diversity.
   * @return A trained [[DecisionTree]] instance.
   */
  def fit(config: TreeConfig, criterion: SplitCriteria, x: DenseMatrix[Double], y: DenseVector[Double], random: Random = new Random()): DecisionTree = {
    val indices = (0 until x.rows).toArray
    val rootNode = buildTree(x, y, indices, 0, config, criterion, random)
    DecisionTree(rootNode, config, criterion)
  }

  /**
   * Trains a Decision Tree model using a specific subset of the provided feature matrix.
   *
   * This overload is used by [[RandomForest]] to implement bagging (bootstrap aggregating)
   * without copying the underlying data matrices.
   *
   * @param config   The configuration parameters for the tree.
   * @param criterion The impurity measure to use for splitting.
   * @param x        The feature matrix.
   * @param y        The target vector.
   * @param indices  An array of indices representing the bootstrap sample.
   * @param random   Random instance for feature sampling.
   * @return A trained [[DecisionTree]] instance.
   */
  def fit(config: TreeConfig, criterion: SplitCriteria, x: DenseMatrix[Double], y: DenseVector[Double], indices: Array[Int], random: Random): DecisionTree = {
    val rootNode = buildTree(x, y, indices, 0, config, criterion, random)
    DecisionTree(rootNode, config, criterion)
  }

  private def buildTree(
    x: DenseMatrix[Double],
    y: DenseVector[Double],
    indices: Array[Int],
    depth: Int,
    config: TreeConfig,
    criterion: SplitCriteria,
    random: Random
  ): Node = {
    val labels = indices.map(i => y(i))
    val numSamples = indices.length
    val numFeatures = x.cols

    // Base cases: max depth, min samples, or pure node
    if (depth >= config.maxDepth || numSamples <= config.minSamplesPerLeaf || (numSamples > 0 && labels.forall(_ == labels(0)))) {
      LeafNode(criterion.calculateLeafValue(labels))
    } else {
      val currentImpurity = criterion.computeImpurity(labels)
      val sampledFeatures = sampleFeatures(numFeatures, config, criterion, random)

      val bestSplit = sampledFeatures.flatMap { fIdx =>
        val sortedIndices = indices.sortBy(i => x(i, fIdx))
        val sortedValues = sortedIndices.map(i => x(i, fIdx))

        (0 until sortedIndices.length - 1).collect {
          case i if sortedValues(i) != sortedValues(i + 1) =>
            val threshold = (sortedValues(i) + sortedValues(i + 1)) / 2.0
            val leftIdx = sortedIndices.slice(0, i + 1)
            val rightIdx = sortedIndices.slice(i + 1, sortedIndices.length)

            val leftLabels = leftIdx.map(idx => y(idx))
            val rightLabels = rightIdx.map(idx => y(idx))

            val weightLeft = leftLabels.length.toDouble / numSamples
            val weightRight = rightLabels.length.toDouble / numSamples

            val gain = currentImpurity - (weightLeft * criterion.computeImpurity(leftLabels) +
                                          weightRight * criterion.computeImpurity(rightLabels))
            Split(gain, fIdx, threshold, leftIdx, rightIdx)
        }
      }

      val bestSplitOption = if (bestSplit.isEmpty) None else Some(bestSplit.maxBy(_.gain))

      bestSplitOption match {
        case Some(split) =>
          InternalNode(
            split.featureIdx,
            split.threshold,
            buildTree(x, y, split.leftIndices, depth + 1, config, criterion, random),
            buildTree(x, y, split.rightIndices, depth + 1, config, criterion, random)
          )
        case None => LeafNode(criterion.calculateLeafValue(labels))
      }
    }
  }

  private def sampleFeatures(numFeatures: Int, config: TreeConfig, criterion: SplitCriteria, random: Random): List[Int] = {
    val featureIndices = (0 until numFeatures).toList
    val count = config.maxFeatures match {
      case Some(n) => n
      case None =>
        if (criterion.isRegression) {
          numFeatures / 3
        } else {
          Math.sqrt(numFeatures).toInt
        }
    }
    random.shuffle(featureIndices).take(math.max(1, count))
  }
}
