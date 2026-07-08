# Product

## Register

product

## Users

Engineers, interviewers, and reviewers watching a live fraud-detection pipeline
demo on a laptop. They open the dashboard, start/adjust the transaction
simulator, and watch scoring behavior and latency in real time. Sessions are
short (minutes), attention is on the numbers moving, and the surface must read
instantly without onboarding.

## Product Purpose

PulseGuard is a production-style real-time fraud detection pipeline (RabbitMQ →
Redis feature store → XGBoost/ONNX → WebSocket). The dashboard is its proof:
it must make throughput, latency percentiles, fraud catches, and live recall
legible at a glance and convincing as an ops surface. Success = a viewer
believes a competent fraud team could actually run on this screen.

## Brand Personality

Mission-control. Serious, dense, precise. An ops room at night: dark surface,
tabular numerics, calm neutrals, with color spent almost exclusively on the
fraud signal and system health. Confidence through restraint, not decoration.

## Anti-references

- The generic AI dashboard: identical stat-card grids, gradient text,
  purple-on-dark SaaS template, glassmorphism, glowing everything.
- Marketing-grade analytics landing pages; this is a working console, not a
  pitch.

## Design Principles

1. **The signal is red, everything else is quiet.** Fraud events are the one
   thing that may shout; all other color is muted and functional.
2. **Numbers first.** Tabular figures, aligned columns, stable layouts; nothing
   jumps or reflows as data streams in.
3. **Density with hierarchy.** Ops surfaces are dense, but every zone has a
   clear rank: KPIs → trends → event streams.
4. **Live means visibly live.** Connection state, motion of incoming rows, and
   ticking charts communicate the pipeline is running without gimmicks.
5. **One screen, no scroll-hunting.** The demo story must be tellable from a
   single viewport at laptop size.

## Accessibility & Inclusion

Sensible defaults: WCAG AA contrast on text, `prefers-reduced-motion`
alternatives for all animation, keyboard-reachable controls, no
color-only encoding of fraud state (icons/labels accompany color).
