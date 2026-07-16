package pt.mleiria.core.nn

import breeze.linalg.DenseVector

/**
 * Holds the intermediate states of an LSTM cell at a specific time step t.
 * These values are necessary for computing gradients during BPTT.
 */
case class LSTMState(
  h: DenseVector[Double],    // Hidden state at time t
  c: DenseVector[Double],    // Cell state at time t
  f: DenseVector[Double],    // Forget gate activation
  i: DenseVector[Double],    // Input gate activation
  cTilde: DenseVector[Double], // Cell candidate activation
  o: DenseVector[Double],    // Output gate activation
  zf: DenseVector[Double],   // Pre-activation for forget gate
  zi: DenseVector[Double],   // Pre-activation for input gate
  zc: DenseVector[Double],   // Pre-activation for cell candidate
  zo: DenseVector[Double]     // Pre-activation for output gate
)
