package dev.hishaam.pulseguard.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "pulseguard")
public record PulseGuardProperties(
        String modelPath,
        String streamPath,
        double fraudThreshold,
        Simulator simulator) {

    public record Simulator(boolean autostart, int tps) {
    }
}
