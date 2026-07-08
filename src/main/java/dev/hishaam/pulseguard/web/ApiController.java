package dev.hishaam.pulseguard.web;

import dev.hishaam.pulseguard.domain.ScoredTransaction;
import dev.hishaam.pulseguard.metrics.PipelineMetrics;
import dev.hishaam.pulseguard.pipeline.TransactionSimulator;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ApiController {

    private final PipelineMetrics metrics;
    private final TransactionSimulator simulator;

    public ApiController(PipelineMetrics metrics, TransactionSimulator simulator) {
        this.metrics = metrics;
        this.simulator = simulator;
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return metrics.snapshot();
    }

    @GetMapping("/transactions/recent")
    public List<ScoredTransaction> recent() {
        return metrics.recentTransactions();
    }

    @PostMapping("/simulator/start")
    public Map<String, Object> start(@RequestParam(defaultValue = "50") int tps) {
        simulator.start(tps);
        return simulatorStatus();
    }

    @PostMapping("/simulator/stop")
    public Map<String, Object> stop() {
        simulator.stop();
        return simulatorStatus();
    }

    @GetMapping("/simulator")
    public Map<String, Object> simulatorStatus() {
        return Map.of(
                "running", simulator.isRunning(),
                "tps", simulator.currentTps(),
                "published", simulator.publishedCount());
    }
}
