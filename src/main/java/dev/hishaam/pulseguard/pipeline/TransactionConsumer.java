package dev.hishaam.pulseguard.pipeline;

import dev.hishaam.pulseguard.config.PulseGuardProperties;
import dev.hishaam.pulseguard.config.RabbitConfig;
import dev.hishaam.pulseguard.domain.CardFeatures;
import dev.hishaam.pulseguard.domain.ScoredTransaction;
import dev.hishaam.pulseguard.domain.Transaction;
import dev.hishaam.pulseguard.features.FeatureStore;
import dev.hishaam.pulseguard.metrics.PipelineMetrics;
import dev.hishaam.pulseguard.scoring.FraudScorer;
import java.util.concurrent.atomic.AtomicLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Component;

/**
 * The scoring stage of the pipeline: consumes raw transactions from RabbitMQ, enriches them with
 * rolling features from Redis, runs XGBoost inference, and fans results out to metrics and the
 * WebSocket dashboard.
 */
@Component
public class TransactionConsumer {

  private static final Logger log = LoggerFactory.getLogger(TransactionConsumer.class);

  /** Cap on non-fraud dashboard broadcasts, so the browser survives high TPS. */
  private static final long BROADCAST_INTERVAL_MILLIS = 100;

  private final FeatureStore featureStore;
  private final FraudScorer scorer;
  private final PipelineMetrics metrics;
  private final SimpMessagingTemplate broker;
  private final double fraudThreshold;
  private final AtomicLong lastBroadcast = new AtomicLong();

  public TransactionConsumer(
      FeatureStore featureStore,
      FraudScorer scorer,
      PipelineMetrics metrics,
      SimpMessagingTemplate broker,
      PulseGuardProperties props) {
    this.featureStore = featureStore;
    this.scorer = scorer;
    this.metrics = metrics;
    this.broker = broker;
    this.fraudThreshold = props.fraudThreshold();
  }

  @RabbitListener(queues = RabbitConfig.QUEUE, concurrency = "4-8")
  public void onTransaction(Transaction txn) {
    long start = System.nanoTime();
    long nowMillis = System.currentTimeMillis();

    CardFeatures features =
        featureStore.readAndUpdate(txn.cardId(), txn.id(), txn.amount(), nowMillis);
    long afterFeatures = System.nanoTime();

    double probability = scorer.score(txn, features);
    long afterInference = System.nanoTime();

    double e2eMillis = (afterInference - txn.producedAtNanos()) / 1_000_000.0;
    // Guard against clock skew if producer and consumer run in different JVMs.
    if (e2eMillis < 0 || e2eMillis > 60_000) {
      e2eMillis = (afterInference - start) / 1_000_000.0;
    }

    ScoredTransaction scored =
        new ScoredTransaction(
            txn.id(),
            txn.cardId(),
            txn.amount(),
            probability,
            probability >= fraudThreshold,
            txn.label(),
            features.txnCount60s(),
            features.timeSinceLastSec(),
            features.amountRatio(),
            (afterFeatures - start) / 1_000.0,
            (afterInference - afterFeatures) / 1_000.0,
            e2eMillis,
            nowMillis);

    metrics.record(scored);
    broadcast(scored);
  }

  private void broadcast(ScoredTransaction scored) {
    if (scored.flagged()) {
      broker.convertAndSend("/topic/frauds", scored);
      broker.convertAndSend("/topic/transactions", scored);
      log.info(
          "FRAUD flagged: txn={} card={} amount={} p={}",
          scored.id(),
          scored.cardId(),
          scored.amount(),
          String.format("%.3f", scored.fraudProbability()));
      return;
    }
    long now = System.currentTimeMillis();
    long last = lastBroadcast.get();
    if (now - last >= BROADCAST_INTERVAL_MILLIS && lastBroadcast.compareAndSet(last, now)) {
      broker.convertAndSend("/topic/transactions", scored);
    }
  }
}
