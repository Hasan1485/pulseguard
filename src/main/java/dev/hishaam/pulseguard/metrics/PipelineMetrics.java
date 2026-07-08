package dev.hishaam.pulseguard.metrics;

import dev.hishaam.pulseguard.domain.ScoredTransaction;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Component;

/**
 * In-memory pipeline metrics: counters, sliding-window throughput and latency
 * reservoirs (last {@value #RESERVOIR_SIZE} samples) with percentile snapshots.
 */
@Component
public class PipelineMetrics {

    private static final int RESERVOIR_SIZE = 4096;
    private static final int RECENT_CAPACITY = 50;

    private final AtomicLong processed = new AtomicLong();
    private final AtomicLong flagged = new AtomicLong();
    private final AtomicLong trueFrauds = new AtomicLong();
    private final AtomicLong fraudsCaught = new AtomicLong();

    private final double[] e2eMillis = new double[RESERVOIR_SIZE];
    private final double[] inferenceMicros = new double[RESERVOIR_SIZE];
    private final double[] featureMicros = new double[RESERVOIR_SIZE];
    private final AtomicInteger reservoirIndex = new AtomicInteger();
    private final AtomicInteger reservoirFill = new AtomicInteger();

    private final ArrayDeque<Long> timestamps = new ArrayDeque<>();
    private final ArrayDeque<ScoredTransaction> recent = new ArrayDeque<>();

    public void record(ScoredTransaction txn) {
        processed.incrementAndGet();
        if (txn.flagged()) {
            flagged.incrementAndGet();
        }
        if (txn.label() == 1) {
            trueFrauds.incrementAndGet();
            if (txn.flagged()) {
                fraudsCaught.incrementAndGet();
            }
        }
        int i = Math.floorMod(reservoirIndex.getAndIncrement(), RESERVOIR_SIZE);
        e2eMillis[i] = txn.e2eMillis();
        inferenceMicros[i] = txn.inferenceMicros();
        featureMicros[i] = txn.featureMicros();
        reservoirFill.updateAndGet(f -> Math.min(f + 1, RESERVOIR_SIZE));

        long now = System.currentTimeMillis();
        synchronized (timestamps) {
            timestamps.addLast(now);
            while (!timestamps.isEmpty() && now - timestamps.peekFirst() > 10_000) {
                timestamps.removeFirst();
            }
        }
        synchronized (recent) {
            recent.addFirst(txn);
            while (recent.size() > RECENT_CAPACITY) {
                recent.removeLast();
            }
        }
    }

    public List<ScoredTransaction> recentTransactions() {
        synchronized (recent) {
            return new ArrayList<>(recent);
        }
    }

    public Map<String, Object> snapshot() {
        double throughput;
        synchronized (timestamps) {
            long now = System.currentTimeMillis();
            while (!timestamps.isEmpty() && now - timestamps.peekFirst() > 10_000) {
                timestamps.removeFirst();
            }
            throughput = timestamps.size() / 10.0;
        }
        long total = processed.get();
        long frauds = trueFrauds.get();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("processed", total);
        map.put("flagged", flagged.get());
        map.put("trueFrauds", frauds);
        map.put("fraudsCaught", fraudsCaught.get());
        map.put("liveRecall", frauds == 0 ? null : (double) fraudsCaught.get() / frauds);
        map.put("throughputTps", throughput);
        map.put("e2eMillis", percentiles(e2eMillis));
        map.put("inferenceMicros", percentiles(inferenceMicros));
        map.put("featureMicros", percentiles(featureMicros));
        return map;
    }

    private Map<String, Double> percentiles(double[] reservoir) {
        int n = reservoirFill.get();
        if (n == 0) {
            return Map.of();
        }
        double[] copy = Arrays.copyOf(reservoir, n);
        Arrays.sort(copy);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("p50", round(copy[(int) (0.50 * (n - 1))]));
        out.put("p95", round(copy[(int) (0.95 * (n - 1))]));
        out.put("p99", round(copy[(int) (0.99 * (n - 1))]));
        out.put("max", round(copy[n - 1]));
        return out;
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }
}
