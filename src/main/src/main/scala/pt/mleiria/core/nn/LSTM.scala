package pt.mleiria.core.nn

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.pow
import pt.mleiria.utils.NormalizeUtils

/**
 * Represents the weights and biases of an LSTM cell.
 *
 * @param Wf Weight matrix for the forget gate.
 * @param bf Bias vector for the forget gate.
 * @param Wi Weight matrix for the input gate.
 * @param bi Bias vector for the input gate.
 * @param Wc Weight matrix for the cell gate (candidate cell state).
 * @param bc Bias vector for the cell gate.
 * @param Wo Weight matrix for the output gate.
 * @param bo Bias vector for the output gate.
 */
case class LSTMWeights(
  Wf: DenseMatrix[Double], bf: DenseVector[Double], // Forget gate
  Wi: DenseMatrix[Double], bi: DenseVector[Double], // Input gate
  Wc: DenseMatrix[Double], bc: DenseVector[Double], // Cell gate
  Wo: DenseMatrix[Double], bo: DenseVector[Double]  // Output gate
)

object LSTMWeights {
  val defaultHiddenDim = 5 // A reasonable default for hidden layer size

  /**
   * Initializes LSTM weights with small random values.
   *
   * @param inputDim  Dimension of the input vector.
   * @param hiddenDim Dimension of the hidden state and cell state.
   * @return A new LSTMWeights instance with randomly initialized values.
   */
  def random(inputDim: Int, hiddenDim: Int): LSTMWeights = {
    // The input to the gates is [h_{t-1}, x_t], so its dimension is hiddenDim + inputDim
    val combinedDim = hiddenDim + inputDim

    // Initialize weights with small random values (e.g., using a small scale factor)
    val scale = 0.01

    val Wf = DenseMatrix.rand(hiddenDim, combinedDim) * scale
    val bf = DenseVector.rand(hiddenDim) * scale
    val Wi = DenseMatrix.rand(hiddenDim, combinedDim) * scale
    val bi = DenseVector.rand(hiddenDim) * scale
    val Wc = DenseMatrix.rand(hiddenDim, combinedDim) * scale
    val bc = DenseVector.rand(hiddenDim) * scale
    val Wo = DenseMatrix.rand(hiddenDim, combinedDim) * scale
    val bo = DenseVector.rand(hiddenDim) * scale

    new LSTMWeights(Wf, bf, Wi, bi, Wc, bc, Wo, bo)
  }
}

/**
 * Implementation of a Long Short-Term Memory (LSTM) network.
 */
object LSTM {

  /**
   * Performs the forward pass over a sequence of inputs.
   *
   * @param x      The sequence of input vectors.
   * @param weights The LSTM weights and biases.
   * @param h0     Initial hidden state.
   * @param c0     Initial cell state.
   * @return A pair: sequence of hidden states and sequence of internal states.
   */
  def forward(x: Seq[DenseVector[Double]],
              weights: LSTMWeights,
              h0: DenseVector[Double],
              c0: DenseVector[Double]): (Seq[DenseVector[Double]], Seq[LSTMState]) = {

    var hPrev = h0
    var cPrev = c0

    val hSeq = scala.collection.mutable.ArrayBuffer[DenseVector[Double]]()
    val stateSeq = scala.collection.mutable.ArrayBuffer[LSTMState]()

    for (xt <- x) {
      // Concatenate previous hidden state and current input: v_t = [h_{t-1}, x_t]
      val vt = DenseVector(hPrev.toArray ++ xt.toArray)

      // Forget Gate
      val zf = (weights.Wf * vt) + weights.bf
      val f = NormalizeUtils.sigmoid(zf)

      // Input Gate
      val zi = (weights.Wi * vt) + weights.bi
      val i = NormalizeUtils.sigmoid(zi)

      // Cell Candidate
      val zc = (weights.Wc * vt) + weights.bc
      val cTilde = NormalizeUtils.tanh(zc)

      // Cell State
      val c = (f * cPrev) + (i * cTilde)

      // Output Gate
      val zo = (weights.Wo * vt) + weights.bo
      val o = NormalizeUtils.sigmoid(zo)

      // Hidden State
      val h = o * NormalizeUtils.tanh(c)

      hSeq += h
      stateSeq += LSTMState(h, c, f, i, cTilde, o, zf, zi, zc, zo)

      hPrev = h
      cPrev = c
    }

    (hSeq.toSeq, stateSeq.toSeq)
  }

  /**
   * Computes the cost (Mean Squared Error) between predictions and targets over a sequence.
   */
  def computeCost(predictions: Seq[DenseVector[Double]],
                  targets: Seq[DenseVector[Double]]): Double = {
    var totalCost = 0.0
    val n = predictions.length

    for (i <- 0 until n) {
      val diff = predictions(i) - targets(i)
      totalCost += sum(pow(diff, 2))
    }

    totalCost / (2.0 * n)
  }

  /**
   * Performs Backpropagation Through Time (BPTT) to compute gradients for weights and biases.
   *
   * @param x      The input sequence.
   * @param targets The target sequence.
   * @param states The sequence of LSTMState from the forward pass.
   * @param weights The current LSTM weights.
   * @return The gradients for the weights and biases in the form of LSTMWeights.
   */
  def backward(x: Seq[DenseVector[Double]],
               targets: Seq[DenseVector[Double]],
               states: Seq[LSTMState],
               weights: LSTMWeights): LSTMWeights = {

    val tMax = x.length
    val hiddenDim = weights.Wf.rows
    val inputDim = x(0).length

    // Accumulators for gradients
    var dWf = DenseMatrix.zeros[Double](hiddenDim, hiddenDim + inputDim)
    var dbf = DenseVector.zeros[Double](hiddenDim)
    var dWi = DenseMatrix.zeros[Double](hiddenDim, hiddenDim + inputDim)
    var dbi = DenseVector.zeros[Double](hiddenDim)
    var dWc = DenseMatrix.zeros[Double](hiddenDim, hiddenDim + inputDim)
.
    var dbc = DenseVector.zeros[Double](hiddenDim)
    var dWo = DenseMatrix.zeros[Double](hiddenDim, hiddenDim + inputdim)
    var dbo = DenseVector.zeros[Double](hiddenDim)

    var dhNext = DenseVector.zeros[Double](hiddenDim)
    var dCNext = DenseVector.zeros[Double](hiddenDim)

    // Iterate backwards from T down to 1
    for (t <- (tMax - 1) to 0 by -1) {
      val state = states(t)
      val xt = x(t)
      val hPrev = if (t == 0) DenseVector.zeros[Double](hiddenDim) else states(t-1).h
      val vt = DenseVector(hPrev.toArray ++ xt.toArray)

      // Loss gradient at time t: dLoss/dh_t
      val dy = state.h - targets(t)
      val dh = dy + dhNext

      // Output gate gradients
      val dOutput = (dh * NormalizeUtils.tanh(state.c) * (state.o * (1.0 - state.o)))
      val dWo_t = dOutput * vt.t
      dbo += dOutput
      dWo += dWo_t

      // Cell state gradients
      val dC = ((dh * state.o * (1.0 - pow(NormalizeUtils.tanh(state.c), 2))) + dCNext)

      // Candidate state gradients
      val dcTilde = (dC * state.i * (1.0 - pow(state.cTilde, 2)))
      val dWc_t = dcTilde * vt.t
      dbc += dcTilde
      dWc += dWc_t

      // Input gate gradients
      val di = (dC * state.cTilde * (state.i * (1.0 - state.i)))
      val dWi_t = di * vt.t
      dbi += di
      dWi += dWi_t

      // Forget gate gradients
      val hPrevC = if (t == 0) DenseVector.zeros[Double](hiddenDim) else states(t-1).c
      val df = (dC * hPrevC * (state.f * (1.0 - state.f)))
      val dWf_t = df * vt.t
      dbf += df
      dWf += dWf_t

      // Pass gradients back to t-1
      val dV = (weights.Wf.t * df) + (weights.Wi.t * di) + (weights.Wc.t * dcTilde) + (weights.Wo.t * dOutput)
      dhNext = dV(0 until hiddenDim)
      dCNext = (state.f * dC)
    }

    LSTMWeights(dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo)
  }
}