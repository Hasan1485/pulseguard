package dev.hishaam.pulseguard.domain;

/**
 * Rolling behavioral features for a card, read from the Redis feature store <em>before</em> the
 * current transaction is folded in (matching offline training).
 */
public record CardFeatures(double txnCount60s, double timeSinceLastSec, double amountRatio) {}
