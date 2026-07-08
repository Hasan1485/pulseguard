package dev.hishaam.pulseguard.domain;

/** A transaction after feature enrichment and model scoring. */
public record ScoredTransaction(
    String id,
    int cardId,
    double amount,
    double fraudProbability,
    boolean flagged,
    int label,
    double txnCount60s,
    double timeSinceLastSec,
    double amountRatio,
    double featureMicros,
    double inferenceMicros,
    double e2eMillis,
    long scoredAtMillis) {}
