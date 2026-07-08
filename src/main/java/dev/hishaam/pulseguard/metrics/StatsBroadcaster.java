package dev.hishaam.pulseguard.metrics;

import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/** Pushes a metrics snapshot to the dashboard once a second. */
@Component
public class StatsBroadcaster {

    private final PipelineMetrics metrics;
    private final SimpMessagingTemplate broker;

    public StatsBroadcaster(PipelineMetrics metrics, SimpMessagingTemplate broker) {
        this.metrics = metrics;
        this.broker = broker;
    }

    @Scheduled(fixedRate = 1000)
    public void broadcastStats() {
        broker.convertAndSend("/topic/stats", (Object) metrics.snapshot());
    }
}
