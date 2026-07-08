package dev.hishaam.pulseguard.scoring;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import dev.hishaam.pulseguard.domain.CardFeatures;
import dev.hishaam.pulseguard.domain.Transaction;
import java.util.Map;
import org.springframework.stereotype.Component;

/**
 * Runs the exported XGBoost model (ONNX) over the 32-feature vector: [Amount, V1..V28,
 * txn_count_60s, time_since_last, amount_ratio].
 */
@Component
public class FraudScorer {

  public static final int NUM_FEATURES = 32;

  private final OrtEnvironment env;
  private final OrtSession session;
  private final String inputName;

  public FraudScorer(OrtEnvironment env, OrtSession session) throws OrtException {
    this.env = env;
    this.session = session;
    this.inputName = session.getInputNames().iterator().next();
  }

  /** Returns the model's probability that this transaction is fraudulent. */
  public double score(Transaction txn, CardFeatures features) {
    float[][] input = new float[1][NUM_FEATURES];
    input[0][0] = (float) txn.amount();
    double[] v = txn.v();
    for (int i = 0; i < 28; i++) {
      input[0][1 + i] = (float) v[i];
    }
    input[0][29] = (float) features.txnCount60s();
    input[0][30] = (float) features.timeSinceLastSec();
    input[0][31] = (float) features.amountRatio();

    try (OnnxTensor tensor = OnnxTensor.createTensor(env, input);
        OrtSession.Result result = session.run(Map.of(inputName, tensor))) {
      // Output 0: label, output 1: probabilities [N, 2] (ZipMap stripped at export)
      float[][] probabilities = (float[][]) result.get(1).getValue();
      return probabilities[0][1];
    } catch (OrtException e) {
      throw new IllegalStateException("ONNX inference failed", e);
    }
  }
}
