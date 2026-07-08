package dev.hishaam.pulseguard.domain;

/**
 * A raw transaction event as published on the RabbitMQ stream.
 *
 * @param producedAtNanos wall-clock capture time (System.currentTimeMillis * 1_000_000
 *                        + nano remainder) used for end-to-end latency measurement
 * @param label ground-truth fraud label carried along for live evaluation (would not
 *              exist in production traffic; -1 when unknown)
 */
public record Transaction(
        String id,
        int cardId,
        double amount,
        double[] v,
        int label,
        long producedAtNanos) {
}
