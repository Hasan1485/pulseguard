package dev.hishaam.pulseguard.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.QueueBuilder;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.JacksonJsonMessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitConfig {

  public static final String EXCHANGE = "pulseguard.tx";
  public static final String QUEUE = "pulseguard.transactions";
  public static final String ROUTING_KEY = "txn";

  @Bean
  DirectExchange transactionExchange() {
    return new DirectExchange(EXCHANGE, true, false);
  }

  @Bean
  Queue transactionQueue() {
    return QueueBuilder.durable(QUEUE).build();
  }

  @Bean
  Binding transactionBinding() {
    return BindingBuilder.bind(transactionQueue()).to(transactionExchange()).with(ROUTING_KEY);
  }

  @Bean
  JacksonJsonMessageConverter jsonMessageConverter() {
    return new JacksonJsonMessageConverter();
  }

  @Bean
  RabbitTemplate rabbitTemplate(
      ConnectionFactory connectionFactory, JacksonJsonMessageConverter converter) {
    RabbitTemplate template = new RabbitTemplate(connectionFactory);
    template.setMessageConverter(converter);
    template.setExchange(EXCHANGE);
    template.setRoutingKey(ROUTING_KEY);
    return template;
  }
}
