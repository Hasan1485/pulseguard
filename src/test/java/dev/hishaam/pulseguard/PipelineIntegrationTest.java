package dev.hishaam.pulseguard;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

import dev.hishaam.pulseguard.domain.ScoredTransaction;
import dev.hishaam.pulseguard.domain.Transaction;
import dev.hishaam.pulseguard.metrics.PipelineMetrics;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;

/**
 * Full-stack pipeline test against real Redis and RabbitMQ (Testcontainers): publishes raw
 * transactions, waits for them to be consumed, feature-enriched, and scored by the ONNX model, then
 * verifies scores, feature-store state, and latency accounting.
 */
@SpringBootTest
class PipelineIntegrationTest {

  @SuppressWarnings("resource")
  static final GenericContainer<?> RABBIT =
      new GenericContainer<>("rabbitmq:3.13-alpine")
          .withEnv("RABBITMQ_DEFAULT_USER", "pulseguard")
          .withEnv("RABBITMQ_DEFAULT_PASS", "pulseguard")
          .withExposedPorts(5672)
          .waitingFor(Wait.forListeningPort())
          .withStartupTimeout(Duration.ofSeconds(90));

  @SuppressWarnings("resource")
  static final GenericContainer<?> REDIS =
      new GenericContainer<>("redis:7-alpine").withExposedPorts(6379);

  static {
    RABBIT.start();
    REDIS.start();
  }

  @DynamicPropertySource
  static void properties(DynamicPropertyRegistry registry) {
    registry.add("spring.rabbitmq.host", RABBIT::getHost);
    registry.add("spring.rabbitmq.port", () -> RABBIT.getMappedPort(5672));
    registry.add("spring.rabbitmq.username", () -> "pulseguard");
    registry.add("spring.rabbitmq.password", () -> "pulseguard");
    registry.add("spring.data.redis.host", REDIS::getHost);
    registry.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
    registry.add("pulseguard.simulator.autostart", () -> "false");
  }

  @Autowired RabbitTemplate rabbitTemplate;

  @Autowired PipelineMetrics metrics;

  @Autowired StringRedisTemplate redisTemplate;

  @Test
  void transactionsFlowThroughThePipelineAndGetScored() {
    int count = 200;
    int cardId = 4242;
    Random random = new Random(7);
    for (int i = 0; i < count; i++) {
      double[] v = new double[28];
      for (int j = 0; j < 28; j++) {
        v[j] = random.nextGaussian();
      }
      rabbitTemplate.convertAndSend(
          new Transaction(
              "it-" + i, cardId, 10 + random.nextDouble() * 200, v, 0, System.nanoTime()));
    }

    await()
        .atMost(Duration.ofSeconds(30))
        .untilAsserted(
            () ->
                assertThat(metrics.snapshot().get("processed"))
                    .satisfies(
                        p -> assertThat(((Number) p).longValue()).isGreaterThanOrEqualTo(count)));

    // Every scored transaction carries a valid probability and latency numbers.
    List<ScoredTransaction> recent = metrics.recentTransactions();
    assertThat(recent).isNotEmpty();
    for (ScoredTransaction txn : recent) {
      assertThat(txn.fraudProbability()).isBetween(0.0, 1.0);
      assertThat(txn.inferenceMicros()).isPositive();
      assertThat(txn.e2eMillis()).isPositive();
    }

    // The Redis feature store accumulated rolling state for the card.
    Map<Object, Object> stats = redisTemplate.opsForHash().entries("fs:" + cardId + ":s");
    assertThat(stats).containsKeys("cnt", "sum", "last");
    assertThat(Long.parseLong((String) stats.get("cnt"))).isEqualTo(count);

    // With 200 txns in well under a minute, the sliding window filled up,
    // so late transactions must have seen a non-zero 60s count.
    assertThat(recent.stream().mapToDouble(ScoredTransaction::txnCount60s).max().orElse(0))
        .isGreaterThan(0);

    // Latency percentiles are being computed.
    @SuppressWarnings("unchecked")
    Map<String, Double> e2e = (Map<String, Double>) metrics.snapshot().get("e2eMillis");
    assertThat(e2e).containsKeys("p50", "p95", "p99");
  }
}
