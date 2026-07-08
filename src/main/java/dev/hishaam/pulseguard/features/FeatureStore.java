package dev.hishaam.pulseguard.features;

import dev.hishaam.pulseguard.domain.CardFeatures;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.springframework.data.redis.core.RedisOperations;
import org.springframework.data.redis.core.SessionCallback;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/**
 * Redis-backed online feature store.
 *
 * <p>Per card it maintains a 60-second sliding window of transaction timestamps (sorted set) plus
 * lifetime aggregates (hash: count, amount sum, last-seen). All commands for one transaction are
 * pipelined into a single round trip; the window count and aggregates are read <em>before</em> the
 * current transaction is folded in, exactly mirroring how the training features were computed.
 */
@Component
public class FeatureStore {

  public static final double WINDOW_MILLIS = 60_000.0;
  public static final double TSL_CAP_SEC = 3600.0;
  private static final Duration KEY_TTL = Duration.ofHours(2);

  private final StringRedisTemplate redis;

  public FeatureStore(StringRedisTemplate redis) {
    this.redis = redis;
  }

  /**
   * Reads the card's rolling features and then folds the current transaction into the store, in one
   * pipelined round trip.
   */
  public CardFeatures readAndUpdate(int cardId, String txnId, double amount, long tsMillis) {
    String windowKey = "fs:" + cardId + ":w";
    String statsKey = "fs:" + cardId + ":s";

    List<Object> results =
        redis.executePipelined(
            new SessionCallback<Object>() {
              @Override
              @SuppressWarnings({"unchecked", "rawtypes"})
              public Object execute(RedisOperations operations) {
                RedisOperations<String, String> ops = operations;
                // -- reads (state BEFORE this transaction) --
                ops.opsForZSet().removeRangeByScore(windowKey, 0, tsMillis - WINDOW_MILLIS);
                ops.opsForZSet().zCard(windowKey);
                ops.opsForHash().entries(statsKey);
                // -- writes (fold this transaction in) --
                ops.opsForZSet().add(windowKey, txnId, tsMillis);
                ops.opsForHash().increment(statsKey, "cnt", 1L);
                ops.opsForHash().increment(statsKey, "sum", amount);
                ops.opsForHash().put(statsKey, "last", Long.toString(tsMillis));
                ops.expire(windowKey, KEY_TTL);
                ops.expire(statsKey, KEY_TTL);
                return null;
              }
            });

    long windowCount = results.get(1) instanceof Long l ? l : 0L;
    @SuppressWarnings("unchecked")
    Map<String, String> stats = (Map<String, String>) results.get(2);

    double timeSinceLast = TSL_CAP_SEC;
    double amountRatio = 1.0;
    if (stats != null && !stats.isEmpty()) {
      long cnt = Long.parseLong(stats.getOrDefault("cnt", "0"));
      double sum = Double.parseDouble(stats.getOrDefault("sum", "0"));
      long last = Long.parseLong(stats.getOrDefault("last", "0"));
      if (last > 0) {
        timeSinceLast = Math.min((tsMillis - last) / 1000.0, TSL_CAP_SEC);
      }
      if (cnt > 0 && sum > 0) {
        amountRatio = amount / (sum / cnt);
      }
    }
    return new CardFeatures(windowCount, timeSinceLast, amountRatio);
  }
}
