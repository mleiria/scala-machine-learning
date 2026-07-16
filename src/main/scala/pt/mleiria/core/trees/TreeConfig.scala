package pt.mleiria.core.trees

/**
 * Configuration for a single Decision Tree.
 *
 * @param maxDepth Maximum depth of the tree. Prevents overfitting by limiting how deep the tree can grow.
 * @param minSamplesPerLeaf Minimum number of samples required to be at a leaf node.
 *                           Small values can lead to overfitting.
 * @param maxFeatures Maximum number of features to consider when looking for the best split.
 *                    If None, a default based on the task (sqrt for classification, 1/3 for regression) is used.
 */
case class TreeConfig(
  maxDepth: Int = 10,
  minSamplesPerLeaf: Int = 2,
  maxFeatures: Option[Int] = None
)

/**
 * Configuration for the Random Forest ensemble.
 *
 * @param numTrees Number of trees in the forest. More trees generally improve stability and accuracy.
 * @param treeConfig Base configuration for each individual tree in the forest.
 * @param criterion The impurity measure to use for splitting ("gini", "entropy", "mse").
 */
case class RFConfig(
  numTrees: Int = 100,
  treeConfig: TreeConfig = TreeConfig(),
  criterion: String = "gini"
)
