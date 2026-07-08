package dev.hishaam.pulseguard.config;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OnnxConfig {

    private static final Logger log = LoggerFactory.getLogger(OnnxConfig.class);

    @Bean
    OrtEnvironment ortEnvironment() {
        return OrtEnvironment.getEnvironment();
    }

    @Bean(destroyMethod = "close")
    OrtSession ortSession(OrtEnvironment env, PulseGuardProperties props) throws OrtException {
        Path modelPath = Path.of(props.modelPath()).toAbsolutePath();
        log.info("Loading ONNX model from {}", modelPath);
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath.toString(), options);
        log.info("Model loaded. Inputs: {}, outputs: {}", session.getInputNames(), session.getOutputNames());
        return session;
    }
}
