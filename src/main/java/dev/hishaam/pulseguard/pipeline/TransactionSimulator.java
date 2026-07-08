package dev.hishaam.pulseguard.pipeline;

import dev.hishaam.pulseguard.config.PulseGuardProperties;
import dev.hishaam.pulseguard.domain.Transaction;
import jakarta.annotation.PreDestroy;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

/**
 * Replays the held-out transaction stream ({@code data/stream.csv.gz}, produced
 * by training) onto RabbitMQ at a configurable rate, looping forever. Stands in
 * for the upstream payment switch that would feed a real deployment.
 */
@Component
public class TransactionSimulator {

    private static final Logger log = LoggerFactory.getLogger(TransactionSimulator.class);

    private final RabbitTemplate rabbitTemplate;
    private final PulseGuardProperties props;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicInteger tps = new AtomicInteger();
    private final AtomicLong published = new AtomicLong();
    private volatile List<Transaction> stream = List.of();

    public TransactionSimulator(RabbitTemplate rabbitTemplate, PulseGuardProperties props) {
        this.rabbitTemplate = rabbitTemplate;
        this.props = props;
        this.tps.set(props.simulator().tps());
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onReady() {
        Path path = Path.of(props.streamPath());
        if (!Files.exists(path)) {
            log.warn("Stream file {} not found; simulator disabled (train the model first)", path);
            return;
        }
        try {
            stream = load(path);
            log.info("Loaded {} transactions from {}", stream.size(), path);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load stream file " + path, e);
        }
        if (props.simulator().autostart()) {
            start(tps.get());
        }
    }

    public synchronized void start(int requestedTps) {
        if (stream.isEmpty()) {
            throw new IllegalStateException("No stream data loaded; run training first");
        }
        tps.set(Math.max(1, Math.min(requestedTps, 2000)));
        if (running.compareAndSet(false, true)) {
            Thread.ofPlatform().name("txn-simulator").daemon(true).start(this::run);
            log.info("Simulator started at {} tps", tps.get());
        } else {
            log.info("Simulator rate changed to {} tps", tps.get());
        }
    }

    public void stop() {
        running.set(false);
        log.info("Simulator stopped");
    }

    public boolean isRunning() {
        return running.get();
    }

    public int currentTps() {
        return tps.get();
    }

    public long publishedCount() {
        return published.get();
    }

    private void run() {
        int index = 0;
        long next = System.nanoTime();
        while (running.get()) {
            Transaction template = stream.get(index);
            index = (index + 1) % stream.size();
            Transaction txn = new Transaction(
                    template.id() + "-" + published.get(),
                    template.cardId(), template.amount(), template.v(), template.label(),
                    System.nanoTime());
            try {
                rabbitTemplate.convertAndSend(txn);
                published.incrementAndGet();
            } catch (Exception e) {
                log.error("Publish failed: {}", e.getMessage());
            }
            next += 1_000_000_000L / tps.get();
            long sleepNanos = next - System.nanoTime();
            if (sleepNanos > 0) {
                try {
                    Thread.sleep(sleepNanos / 1_000_000, (int) (sleepNanos % 1_000_000));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            } else {
                next = System.nanoTime();
            }
        }
    }

    private static List<Transaction> load(Path path) throws IOException {
        List<Transaction> txns = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new GZIPInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8))) {
            String header = reader.readLine(); // card_id,Amount,V1..V28,Class
            if (header == null) {
                return txns;
            }
            String line;
            int row = 0;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                double[] v = new double[28];
                for (int i = 0; i < 28; i++) {
                    v[i] = Double.parseDouble(parts[2 + i]);
                }
                txns.add(new Transaction(
                        "txn-" + row++,
                        Integer.parseInt(parts[0]),
                        Double.parseDouble(parts[1]),
                        v,
                        Integer.parseInt(parts[30]),
                        0L));
            }
        }
        return txns;
    }

    @PreDestroy
    void shutdown() {
        running.set(false);
    }
}
