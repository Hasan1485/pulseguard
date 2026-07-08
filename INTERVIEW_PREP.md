# PulseGuard — Interview Preparation Document

> Real-Time Fraud Detection Pipeline — Spring Boot 4, RabbitMQ, Redis, XGBoost served via ONNX Runtime.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Folder Structure](#architecture--folder-structure)
3. [Concepts & Tooling Learned](#concepts--tooling-learned)
4. [OOP & Spring Principles Used (with Code Examples)](#oop--spring-principles-used-with-code-examples)
5. [Module-by-Module Deep Dive](#module-by-module-deep-dive)
6. [The ML Side: Training, SMOTE, ONNX](#the-ml-side-training-smote-onnx)
7. [Benchmarking: Methodology & Results](#benchmarking-methodology--results)
8. [Common Interview Questions & Answers](#common-interview-questions--answers)
9. [Potential Improvements](#potential-improvements)
10. [Quick Revision Cheatsheet](#quick-revision-cheatsheet)

---

## Project Overview

**PulseGuard** scores payment transactions for fraud in real time. Transactions stream through RabbitMQ; a consumer pool enriches each one with rolling behavioral features from a Redis feature store, runs an XGBoost classifier (exported to ONNX, served in-process from Java), and fans results out to metrics and a live WebSocket feed. The model is trained offline in Python on a real open dataset with SMOTE to handle extreme class imbalance.

### Key Features

- **RabbitMQ streaming pipeline** — durable queue, JSON messages, concurrent listener pool (4–8 consumers, prefetch 64), sustained at 500 txn/s locally.
- **Redis-backed online feature store** — per-card 60-second sliding window (sorted set) + lifetime aggregates (hash), all commands for one transaction pipelined into a **single round trip**.
- **Offline/online feature parity** — the training script computes the exact same rolling features, with the same read-before-write semantics, while replaying the dataset in time order.
- **XGBoost via ONNX Runtime** — trained with SMOTE (577:1 imbalance), exported to ONNX with the ZipMap node stripped, parity-checked against `predict_proba` at export time, served in-process at ~70 µs p50.
- **Live evaluation** — ground-truth labels ride along with replayed transactions, so the pipeline reports *live recall* (frauds caught / true frauds seen) as it runs.
- **Full-stack integration test** — Testcontainers boots real Redis + RabbitMQ, pushes 200 transactions through the entire pipeline, asserts on scores, Redis state, and latency accounting.

### Tech Stack

| Technology | Purpose |
|---|---|
| Java 25 + Spring Boot 4 | Application framework (web, AMQP, Redis, WebSocket starters) |
| RabbitMQ 3.13 | Message broker — the transaction stream |
| Redis 7 | Online feature store (rolling aggregates) |
| ONNX Runtime (Java) | In-process model inference |
| Python 3.13 + XGBoost | Offline training |
| imbalanced-learn (SMOTE) | Class-imbalance handling |
| onnxmltools | XGBoost → ONNX conversion |
| Testcontainers 2 | Integration tests against real containers |
| Docker Compose | Local infra (Redis, RabbitMQ) |
| uv + ruff + ty | Python env/lock, lint/format, type checking |
| Spotless (google-java-format) | Java formatting, enforced in the build |

---

## Architecture & Folder Structure

```
pulseguard/
├── training/
│   ├── train.py            ← dataset download, feature engineering, SMOTE, XGBoost, ONNX export
│   ├── pyproject.toml      ← uv project (deps + ruff/ty dev group)
│   └── uv.lock
├── model/                  ← fraud_xgb.onnx + metrics.json (generated, gitignored)
├── data/                   ← stream.csv.gz replay file (generated, gitignored)
├── src/main/java/dev/hishaam/pulseguard/
│   ├── config/
│   │   ├── RabbitConfig.java          exchange/queue/binding + JSON converter + RabbitTemplate
│   │   ├── OnnxConfig.java            OrtEnvironment + OrtSession beans (model loaded once)
│   │   ├── WebSocketConfig.java       STOMP endpoint /ws, /topic simple broker
│   │   └── PulseGuardProperties.java  typed @ConfigurationProperties record
│   ├── domain/
│   │   ├── Transaction.java           raw event on the stream (record)
│   │   ├── CardFeatures.java          rolling features read from Redis (record)
│   │   └── ScoredTransaction.java     enriched + scored result (record)
│   ├── pipeline/
│   │   ├── TransactionSimulator.java  replays held-out data onto RabbitMQ at N tps
│   │   └── TransactionConsumer.java   @RabbitListener: features → inference → fan-out
│   ├── features/FeatureStore.java     Redis sliding window + aggregates, one pipelined trip
│   ├── scoring/FraudScorer.java       builds float[1][32], runs the ONNX session
│   ├── metrics/
│   │   ├── PipelineMetrics.java       counters, latency reservoirs, percentiles, throughput
│   │   └── StatsBroadcaster.java      1 Hz snapshot push to /topic/stats
│   └── web/ApiController.java         REST: stats, recent txns, simulator control
├── src/main/resources/
│   ├── application.yml                env-overridable config (hosts, tps, threshold, paths)
│   └── static/index.html              dashboard (STOMP client; not covered here)
├── src/test/java/.../PipelineIntegrationTest.java   Testcontainers full-pipeline test
├── docker-compose.yml                 Redis + RabbitMQ with healthchecks
└── pom.xml                            Boot 4 parent, Testcontainers BOM, Spotless
```

### High-Level Data Flow

```
┌────────────────┐   publish    ┌───────────────┐   consume    ┌────────────────────────────┐
│ Transaction    │ ───────────▶ │   RabbitMQ    │ ───────────▶ │  TransactionConsumer (4-8) │
│ Simulator      │   (JSON)     │ durable queue │  prefetch 64 │                            │
│ (replay N tps) │              └───────────────┘              │  1. FeatureStore.readAnd-  │
└────────────────┘                                             │     Update()  ──▶ Redis    │
                                                               │  2. FraudScorer.score()    │
                                                               │     ──▶ ONNX Runtime       │
                                                               │  3. metrics.record()       │
                                                               │  4. broadcast() ──▶ STOMP  │
                                                               └────────────────────────────┘
                                                                        │
                              /topic/transactions  /topic/frauds  /topic/stats
                                                                        │
                                                               WebSocket subscribers
```

### One Transaction, Step by Step

```
1. Simulator picks the next row of the held-out stream, stamps producedAtNanos,
   publishes JSON to exchange "pulseguard.tx" (routing key "txn").
2. RabbitMQ routes it to durable queue "pulseguard.transactions".
3. A listener thread receives it (Jackson deserializes into the Transaction record).
4. FeatureStore: ONE pipelined Redis round trip —
      reads  : ZREMRANGEBYSCORE (expire window), ZCARD (count), HGETALL (aggregates)
      writes : ZADD (this txn), HINCRBY cnt, HINCRBYFLOAT sum, HSET last, EXPIRE x2
   Features come from the reads → state BEFORE this transaction (training parity).
5. FraudScorer builds float[1][32] = [Amount, V1..V28, count60s, timeSinceLast, amountRatio]
   and runs the ONNX session → P(fraud).
6. Consumer assembles ScoredTransaction with three timings:
   featureMicros, inferenceMicros, e2eMillis (now − producedAtNanos).
7. PipelineMetrics.record() → counters + ring-buffer reservoirs + recent list.
8. broadcast(): flagged txns always go to /topic/frauds; the sampled stream to
   /topic/transactions is rate-limited (~10/s) so high TPS can't flood subscribers.
```

---

## Concepts & Tooling Learned

### Infrastructure & Messaging

| Concept | Explanation | Where in PulseGuard |
|---|---|---|
| **Exchange / queue / binding** | Producer sends to an *exchange*; a *binding* (routing key) decides which *queue* gets it; consumers read the queue | `RabbitConfig`: DirectExchange `pulseguard.tx` → binding `txn` → queue `pulseguard.transactions` |
| **Durable queue** | Queue metadata survives broker restart (messages too, if persistent) | `QueueBuilder.durable(QUEUE)` |
| **Prefetch (QoS)** | How many unacked messages the broker pushes to one consumer at a time; too low = starved consumers, too high = unfair batching | `spring.rabbitmq.listener.simple.prefetch: 64` |
| **Listener concurrency** | Multiple consumer threads on one queue = parallelism without partitioning | `@RabbitListener(concurrency = "4-8")` |
| **Message converter** | Serializes/deserializes payloads; content-type driven | `JacksonJsonMessageConverter` bean used by template + listener |
| **Redis pipelining** | Send N commands in one network round trip, read N replies; no server-side atomicity guarantee, but massive latency win | `redis.executePipelined(SessionCallback)` in `FeatureStore` |
| **Redis ZSET as sliding window** | Members scored by timestamp; `ZREMRANGEBYSCORE` drops expired entries, `ZCARD` counts the rest | `fs:{card}:w` window key |
| **Key TTL discipline** | Feature keys expire (2 h) so an unbounded card population can't grow Redis forever | `EXPIRE` on both keys every write; `maxmemory allkeys-lru` as backstop in compose |
| **Healthchecks in Compose** | `docker compose up` reports healthy only when Redis PONGs and RabbitMQ responds to `rabbitmq-diagnostics ping` | `docker-compose.yml` |

### Spring Boot 4 / Java 25

| Concept | Explanation | Where |
|---|---|---|
| **Records** | Immutable data carriers with generated ctor/equals/hashCode; ideal for messages & DTOs | `Transaction`, `CardFeatures`, `ScoredTransaction`, `PulseGuardProperties` |
| **`@ConfigurationProperties` record** | Typed, immutable config binding instead of `@Value` string plumbing | `PulseGuardProperties(modelPath, streamPath, fraudThreshold, simulator)` |
| **Bean lifecycle & `destroyMethod`** | Container closes native resources on shutdown | `@Bean(destroyMethod = "close") OrtSession` |
| **`ApplicationReadyEvent`** | Run logic only after the app is fully up (listeners connected) | Simulator loads the stream + autostarts |
| **`@Scheduled`** | Fixed-rate background work on the scheduler thread | `StatsBroadcaster` 1 Hz snapshot |
| **STOMP over WebSocket** | Frame-based pub/sub protocol on a WebSocket; Spring's simple broker manages `/topic/*` fan-out | `WebSocketConfig` + `SimpMessagingTemplate` |
| **Virtual/platform threads API** | `Thread.ofPlatform().daemon(true)` builder replaces raw `new Thread()` | Simulator publisher thread |
| **Boot 3 → 4 migration** | `spring-boot-starter-web` → `-webmvc`; `Jackson2JsonMessageConverter` (Jackson 2) deprecated for `JacksonJsonMessageConverter` (Jackson 3) | pom + `RabbitConfig` |

### ML & Python Tooling

| Concept | Explanation | Where |
|---|---|---|
| **Class imbalance** | 492 frauds in 284,807 rows (~577:1); accuracy is useless, recall/PR-AUC matter | dataset; metrics.json |
| **SMOTE** | Synthetic Minority Over-sampling TEchnique — interpolates new minority samples between real neighbors instead of duplicating | `SMOTE().fit_resample(X_tr, y_tr)` on the **train split only** |
| **Stratified split** | Keeps the fraud ratio identical in train/test | `train_test_split(..., stratify=y)` |
| **ROC-AUC vs PR-AUC** | ROC-AUC can look great under imbalance; PR-AUC is the honest curve when positives are rare | both reported |
| **ONNX** | Portable model graph format; lets Python-trained models run in Java/C++/anywhere | `convert_xgboost(...)` |
| **ZipMap stripping** | Classifier converters emit probabilities as a sequence of maps (awkward in Java); removing the ZipMap node exposes a plain float [N,2] tensor | graph surgery in `export_onnx()` |
| **Export parity check** | ONNX output compared to `predict_proba` before the model is accepted (max abs diff must be < 1e-3; observed 0.0) | `export_onnx()` assertion |
| **uv** | Fast Python package/project manager; `uv sync` creates the venv from `pyproject.toml` + lockfile | `training/` |
| **ruff / ty** | Linter+formatter and type checker (both Astral); both must pass clean | `uv run ruff check`, `uv run ty check` |

---

## OOP & Spring Principles Used (with Code Examples)

### 1. Dependency Injection / Inversion of Control

**Definition**: Classes declare what they need in the constructor; the Spring container builds the object graph and hands dependencies in. Nothing constructs its own collaborators.

```java
@Component
public class TransactionConsumer {
  private final FeatureStore featureStore;
  private final FraudScorer scorer;
  private final PipelineMetrics metrics;
  private final SimpMessagingTemplate broker;

  public TransactionConsumer(FeatureStore featureStore, FraudScorer scorer,
      PipelineMetrics metrics, SimpMessagingTemplate broker, PulseGuardProperties props) {
    ...
  }
}
```

**Benefit**: in the integration test the same consumer runs unchanged against Testcontainers-provided Redis/RabbitMQ — only properties change, no code.

**Where**: every component; no field injection, no `@Autowired` on constructors (implicit single-constructor injection).

---

### 2. Immutability via Records

**Definition**: Data that flows between threads is immutable — no setters, no shared mutable state.

```java
public record Transaction(
    String id, int cardId, double amount, double[] v, int label, long producedAtNanos) {}

public record ScoredTransaction(
    String id, int cardId, double amount, double fraudProbability, boolean flagged, int label,
    double txnCount60s, double timeSinceLastSec, double amountRatio,
    double featureMicros, double inferenceMicros, double e2eMillis, long scoredAtMillis) {}
```

**Why it matters here**: messages cross threads constantly (publisher thread → broker → 4–8 listener threads → scheduler thread reading metrics). Records make that safe by construction; Jackson also (de)serializes them without ceremony.

---

### 3. Single Responsibility — pipeline stages as separate components

Each stage of the pipeline is one class with one job:

| Class | Responsibility |
|---|---|
| `TransactionSimulator` | produce the stream |
| `TransactionConsumer` | orchestrate one message end to end |
| `FeatureStore` | Redis state, feature semantics |
| `FraudScorer` | tensor building + inference |
| `PipelineMetrics` | counting, reservoirs, percentiles |
| `StatsBroadcaster` | periodic snapshot push |

The consumer is the *orchestrator*: it owns the order and timing measurement, but delegates every domain concern.

---

### 4. Configuration as a Typed, Immutable Object

```java
@ConfigurationProperties(prefix = "pulseguard")
public record PulseGuardProperties(
    String modelPath, String streamPath, double fraudThreshold, Simulator simulator) {
  public record Simulator(boolean autostart, int tps) {}
}
```

One bean, bound and validated at startup, injected wherever needed. Environment variables override every field (`${MODEL_PATH:model/fraud_xgb.onnx}` etc. in `application.yml`) — the same jar runs locally, in Docker, or in tests without edits.

---

### 5. Publish/Subscribe (Observer) — twice

Two different pub/sub systems, used deliberately at different tiers:

- **RabbitMQ** decouples the producer from the scoring consumers (durable, backpressure-aware, competing consumers).
- **STOMP topics** decouple the pipeline from any number of live subscribers (`/topic/transactions`, `/topic/frauds`, `/topic/stats`) — subscribers can come and go; the pipeline never knows about them.

---

### 6. Resource Management Around Native Code

ONNX Runtime is a native library — tensors and sessions are off-heap and must be closed:

```java
try (OnnxTensor tensor = OnnxTensor.createTensor(env, input);
    OrtSession.Result result = session.run(Map.of(inputName, tensor))) {
  float[][] probabilities = (float[][]) result.get(1).getValue();
  return probabilities[0][1];
}
```

try-with-resources per call; the long-lived `OrtSession` is closed by the container (`destroyMethod = "close"`). `OrtSession.run()` is thread-safe, so 4–8 listener threads share one session — the model is loaded into memory exactly once.

---

### 7. Lock-Free Where It Counts, Coarse Locks Where It Doesn't

`PipelineMetrics` mixes two strategies deliberately:

- **Hot path** (every transaction): `AtomicLong` counters and a fixed ring buffer indexed by `AtomicInteger.getAndIncrement() % SIZE` — no locks, writes race benignly (a slot may be overwritten; a reservoir tolerates that by design).
- **Cold path** (bounded deques for throughput timestamps and recent transactions): small `synchronized` blocks — contention is trivial at these sizes and correctness is simpler to reason about.

---

## Module-by-Module Deep Dive

### features/FeatureStore — The Online Feature Store

**Redis data model, per card:**

```
fs:{cardId}:w   ZSET   member = txnId, score = eventTimeMillis     (sliding window)
fs:{cardId}:s   HASH   cnt = lifetime count, sum = lifetime amount, last = last-seen ms
```

**One pipelined round trip per transaction** (order matters):

```
reads  (state BEFORE this txn):        writes (fold this txn in):
1. ZREMRANGEBYSCORE  w  0 (t-60s)      4. ZADD      w  txnId t
2. ZCARD             w                 5. HINCRBY   s  cnt 1
3. HGETALL           s                 6. HINCRBYFLOAT s sum amount
                                       7. HSET      s  last t
                                       8. EXPIRE    w  2h
                                       9. EXPIRE    s  2h
```

**The three features** (identical semantics to training):

| Feature | Definition | Default (first txn) |
|---|---|---|
| `txn_count_60s` | # of the card's txns in the last 60 s, **excluding this one** | 0 |
| `time_since_last` | seconds since the card's previous txn, capped at 3600 | 3600 |
| `amount_ratio` | this amount ÷ card's lifetime mean amount | 1.0 |

**Why read-before-write?** The model was trained on features computed from each card's *prior* history. If serving included the current transaction in the window count, every feature would be off by one relative to training — a subtle train/serve skew that silently degrades the model. This is the classic "offline/online feature parity" problem that real feature stores (Feast, Tecton) exist to solve.

**Why pipelining and not a Lua script / MULTI?** Pipelining gives the latency win (1 round trip instead of 9) without server-side scripting. The read/write split has a benign race: two concurrent txns for the *same card* could both read the same "before" state. At worst a window count is off by one for one scoring — acceptable for features, not worth a Lua script's operational complexity. (Interview-honest answer: if strict per-card serialization mattered, use a Lua script or hash the card to a single consumer.)

---

### scoring/FraudScorer — ONNX Inference

- Feature vector layout is a **fixed contract with training**: `[Amount, V1..V28, txn_count_60s, time_since_last, amount_ratio]` — 32 floats. Order mismatch = silently wrong predictions, which is why the ONNX export is parity-checked.
- Input: `float[1][32]` tensor. Output 0 is the argmax label (ignored), output 1 is `float[N][2]` probabilities; `[0][1]` is P(fraud).
- The flag decision (`p >= threshold`, default 0.5) lives in the consumer via config — the scorer returns a probability, policy stays configurable.

---

### pipeline/TransactionSimulator — The Producer

- Loads `data/stream.csv.gz` (56,962 held-out, time-ordered transactions) into memory once at `ApplicationReadyEvent`; warns and disables itself if training hasn't been run.
- A single daemon platform thread publishes at the requested rate using an **absolute-deadline scheduler**: `next += 1e9 / tps`, sleep the remainder. This avoids drift that per-iteration `sleep(1000/tps)` would accumulate; if the publisher falls behind, it resets the deadline instead of bursting.
- Rate is an `AtomicInteger` — changing TPS via the REST endpoint takes effect on the next tick without restarting the thread. Loops the dataset forever; each replay gets a fresh unique id (`txn-123-<counter>`).
- Stamps `producedAtNanos` (System.nanoTime) at publish for end-to-end latency measurement.

---

### pipeline/TransactionConsumer — The Scoring Stage

- `@RabbitListener(queues = ..., concurrency = "4-8")` — the container scales listener threads between 4 and 8; combined with prefetch 64 this keeps consumers fed at 500 tps.
- Measures three latencies with `System.nanoTime()`:
  - `featureMicros` — the Redis round trip
  - `inferenceMicros` — ONNX session run
  - `e2eMillis` — now − `producedAtNanos` (publish → scored, includes broker time)
- Clock-skew guard: if e2e is negative or absurd (> 60 s — e.g., producer in another JVM with a different nanoTime origin), it falls back to consumer-side processing time. `System.nanoTime()` is only meaningful within one JVM.
- Broadcast policy: **every** flagged transaction goes out on `/topic/frauds`; the general stream on `/topic/transactions` is sampled (at most one per 100 ms, enforced with an `AtomicLong` CAS) so subscribers survive high TPS.

---

### metrics/PipelineMetrics — Latency Accounting

- **Counters**: processed, flagged, trueFrauds, fraudsCaught (`AtomicLong`). Live recall = fraudsCaught / trueFrauds — computable only because ground-truth labels ride along in the replay.
- **Reservoirs**: three fixed `double[4096]` ring buffers (e2e, inference, feature). Index = atomic counter mod size → last 4096 samples, O(1) write, no allocation on the hot path.
- **Percentiles**: on snapshot, copy + sort the filled portion, index at 0.50/0.95/0.99 — O(n log n) once per second on ≤4096 elements, effectively free.
- **Throughput**: a deque of timestamps trimmed to a 10 s window; size ÷ 10 = txn/s. Simple and exact for a dashboard.
- **Snapshot** is a `LinkedHashMap` so JSON field order is stable for consumers.

---

### config/ — Wiring

- **RabbitConfig**: declares exchange/queue/binding as beans (Spring auto-declares them on the broker at startup — idempotent), a Jackson JSON converter shared by template and listener container, and a `RabbitTemplate` pre-bound to the exchange + routing key.
- **OnnxConfig**: one `OrtEnvironment` + one `OrtSession` for the whole app; model path comes from properties; logs input/output names at startup as a sanity check.
- **WebSocketConfig**: STOMP endpoint at `/ws`, in-memory simple broker for `/topic/*`. No external broker needed at this scale; the upgrade path is a relay to RabbitMQ's STOMP plugin.

---

### The Integration Test (Testcontainers)

```java
static final GenericContainer<?> RABBIT = new GenericContainer<>("rabbitmq:3.13-alpine")
    .withEnv("RABBITMQ_DEFAULT_USER", "pulseguard")...
static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7-alpine")...
static { RABBIT.start(); REDIS.start(); }

@DynamicPropertySource
static void properties(DynamicPropertyRegistry registry) {
  registry.add("spring.rabbitmq.host", RABBIT::getHost);
  registry.add("spring.rabbitmq.port", () -> RABBIT.getMappedPort(5672));
  ...
  registry.add("pulseguard.simulator.autostart", () -> "false");
}
```

What it proves, in one test:

1. 200 messages published via `RabbitTemplate` are all consumed and scored (Awaitility polls the metrics until `processed >= 200`).
2. Every scored transaction has a probability in [0,1] and positive latency numbers → the real ONNX model loaded and ran.
3. Redis contains the card's hash with `cnt == 200` → the feature store folded in every transaction exactly once.
4. Late transactions saw a non-zero 60s window count → sliding-window reads work.
5. The snapshot exposes p50/p95/p99 → latency accounting works.

Notes: containers start once per class via a static initializer; ports are dynamic (`getMappedPort`) so tests never collide with local infra; the simulator is disabled so the test controls exactly what enters the pipeline.

---

## The ML Side: Training, SMOTE, ONNX

### Dataset

**ULB Credit Card Fraud** (OpenML dataset 1597, no auth needed via `sklearn.fetch_openml`): 284,807 real (anonymized) European card transactions over 2 days, 492 frauds (0.172%). Features V1–V28 are PCA components of confidential originals; `Amount` is raw. The OpenML export drops the `Time` column but preserves row order, so training reconstructs monotonic event times with seeded exponential inter-arrivals at the dataset's true average rate.

### Feature Engineering with Parity

1. Assign each row a synthetic `card_id` (2,000 cards, Dirichlet-skewed so some cards are hot — like real usage).
2. Replay rows in time order, maintaining per-card state (timestamp deque, count, sum, last-seen).
3. For each row, **read features first** (window count, time-since-last capped at 3600, amount/mean ratio), **then** fold the row into state — the exact algorithm `FeatureStore` runs against Redis.

Final feature vector: 32 = Amount + V1..V28 + 3 behavioral features.

### Handling Imbalance: SMOTE

- Split 80/20 stratified **first**, then SMOTE only on the training split (227,845 rows, 394 frauds → 454,902 balanced rows). Applying SMOTE before splitting would leak synthetic copies of test frauds into training — a classic evaluation bug.
- SMOTE creates synthetic minority points by interpolating between a fraud sample and its k nearest fraud neighbors — better decision boundaries than naive duplication, which just reweights.
- The test set keeps the natural 577:1 imbalance — evaluation must reflect reality.

### Model & Results

`XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, tree_method="hist")`

| Metric (held-out 56,962 rows, 98 frauds) | Value |
|---|---|
| ROC-AUC | **0.978** |
| PR-AUC | 0.883 |
| Recall @ 0.5 | **0.867** (85/98 frauds caught) |
| Precision @ 0.5 | 0.810 |

Threshold 0.5 is a demo default; production would tune it on the precision/recall trade-off (cost of a missed fraud vs. cost of a false alarm) — the pipeline reads it from config.

### ONNX Export Pipeline

```
XGBClassifier ──convert_xgboost──▶ ONNX graph ──strip ZipMap──▶ fraud_xgb.onnx (795 KB)
                                                     │
                              onnxruntime parity check vs predict_proba
                              (max |diff| < 1e-3 required; observed 0.000000)
```

- **Why ONNX instead of calling Python from Java?** No network hop, no second service, no serialization overhead — inference is an in-process function call (~70 µs). The JVM never knows Python existed.
- **Why strip ZipMap?** Classifier converters wrap probabilities in a sequence-of-maps output for scikit-learn compatibility. In Java that's `OnnxSequence` unwrapping per call. Removing the node and rewiring the graph output exposes a plain `float[N][2]` tensor.
- **Why the parity check?** The 32-feature contract and the converted trees must produce identical probabilities. Checking at export time (not in production) means a bad conversion can never ship.

---

## Benchmarking: Methodology & Results

### Methodology

- The pipeline measures itself: every scored transaction contributes to 4096-sample sliding reservoirs; percentiles are computed on snapshot. Numbers below were read from `/api/stats` under sustained load on an Apple Silicon laptop, everything local (app + Redis + RabbitMQ in Docker).
- `e2eMillis` spans publish (`producedAtNanos`) to scored — it includes broker transit, queueing, Redis, and inference.
- Load driver is the built-in simulator (`POST /api/simulator/start?tps=500`).

### Results @ 500 txn/s sustained

| Metric | p50 | p95 | p99 |
|---|---|---|---|
| End-to-end (publish → scored) | 2.5 ms | 4.2 ms | 6.8 ms |
| ONNX inference | 69 µs | 144 µs | 242 µs |
| Redis feature round trip | 1.7 ms | 2.9 ms | 4.9 ms |

Reading the numbers:

1. **The feature store dominates e2e** (~1.7 of 2.5 ms p50) — one network round trip to Redis. Inference is ~40× cheaper than the feature fetch; the bottleneck is I/O, not the model.
2. **Throughput was simulator-limited, not pipeline-limited**: measured 499–500 txn/s at the target rate, with p99 still under 7 ms — the consumers weren't saturated.
3. **Startup transient**: the first seconds show large e2e (messages queued before listeners warmed up); the sliding reservoir flushes it out. Real lesson: measure steady state, note warm-up separately.
4. Live recall under replay (~90%) tracks the offline test recall (86.7%) — sanity that serving matches training.

---

## Common Interview Questions & Answers

### Q1: "Explain the project at a high level."

> It's a real-time fraud scoring pipeline. A simulator replays real credit-card transactions onto RabbitMQ; a pool of Spring consumers enriches each one with rolling behavioral features kept in Redis — count of the card's transactions in the last 60 seconds, time since its last transaction, amount versus its average — then scores it with an XGBoost model served in-process through ONNX Runtime, in about 70 microseconds. Results fan out over WebSocket topics and feed live metrics, including live recall, because ground-truth labels ride along with the replayed traffic. The model is trained offline in Python with SMOTE for the 577:1 class imbalance, and the training script computes the exact same rolling features with the same semantics as the online store — that offline/online parity is the heart of the design.

### Q2: "Why RabbitMQ and not Kafka?"

> For this workload — competing consumers on one work queue, per-message ack, low latency at hundreds per second — a broker queue is the natural fit and operationally lighter. Kafka earns its complexity when you need replay, multiple independent consumer groups reading the same history, partitioned ordering, or six-figure throughput. Nothing in the design blocks a swap: the consumer is one `@RabbitListener` method; the feature-store and scoring stages wouldn't change. If I needed per-card ordering guarantees, Kafka's key-based partitioning would actually be the stronger argument for switching.

### Q3: "Walk me through the Redis feature store design."

> Two keys per card: a sorted set as the sliding window — members are transaction ids, scores are timestamps, so `ZREMRANGEBYSCORE` expires the window and `ZCARD` counts it — and a hash with lifetime count, amount sum, and last-seen. All nine commands for one transaction go in a single pipeline: three reads first, then the writes. The reads deliberately happen before the writes so features reflect the card's state *before* the current transaction — matching exactly how the training features were computed. Both keys carry a 2-hour TTL so the key population stays bounded, with allkeys-lru as a backstop.

### Q4: "What is offline/online feature parity and why does it matter?"

> The model learns feature *semantics* from training data. If serving computes even slightly different semantics — say, including the current transaction in the window count when training excluded it — every prediction happens on a distribution the model never saw. It's called train/serve skew, it doesn't crash anything, and it silently costs you accuracy. I enforced parity by making both sides implement the same algorithm — read state, compute features, then fold the transaction in — and the training script replays the dataset in time order like a stream.

### Q5: "Why serve the model with ONNX Runtime instead of a Python microservice?"

> Latency and operational simplicity. In-process inference is a function call — ~70 µs p50 — versus a network hop, serialization, and a second service to deploy, monitor, and scale. ONNX is the portability layer: train in Python where the ecosystem is, export a graph, serve from Java. The trade-offs: you must freeze a strict feature-vector contract (order and count), and you should verify the conversion — my export step asserts ONNX output matches XGBoost's `predict_proba` before writing the file, and observed an exact match.

### Q6: "What is SMOTE and why not just duplicate the minority class or weight it?"

> SMOTE synthesizes new minority samples by interpolating between a real fraud and its nearest fraud neighbors, so the classifier sees a filled-in minority region rather than the same 394 points repeated. Naive duplication is equivalent to reweighting and tends to overfit those exact points. Class weights (`scale_pos_weight` in XGBoost) are a legitimate alternative — often comparable — but SMOTE was the deliberate choice here. The critical detail either way: resample only the training split, after the split. SMOTE before splitting leaks synthetic copies of test frauds into training and inflates every metric.

### Q7: "Your dataset is 99.8% legitimate. Why is accuracy the wrong metric, and what did you use?"

> Predicting "legit" for everything scores 99.83% accuracy and catches zero fraud. I report recall (fraction of true frauds caught — 0.867 at threshold 0.5), precision (0.81 — how many alarms are real), PR-AUC (0.883 — precision/recall across all thresholds; the honest curve under heavy imbalance), and ROC-AUC (0.978 — threshold-free ranking quality, but optimistic under imbalance, which is why I don't report it alone).

### Q8: "How do you measure end-to-end latency across the pipeline?"

> The producer stamps `System.nanoTime()` into the message at publish; the consumer computes now minus that stamp after scoring — so e2e includes broker transit, queue wait, Redis, and inference. Within the consumer I also bracket the Redis call and the ONNX call separately, which is how I know the feature fetch (~1.7 ms) dominates and inference (~70 µs) is almost free. Caveat I handle explicitly: nanoTime is only comparable within one JVM, so there's a skew guard that falls back to consumer-side timing if the delta is negative or absurd.

### Q9: "How do the latency percentiles work internally?"

> Each latency stream writes into a fixed 4096-slot ring buffer — atomic increment mod size, no locks, no allocation on the hot path, and old samples age out naturally, so it's a sliding window over the last 4096 transactions. Once a second, the snapshot copies the filled portion, sorts it, and indexes at the 50th/95th/99th positions. Sorting ≤4096 doubles once a second is negligible. It's the same reservoir idea Dropwizard/Micrometer histograms use, just deliberately simple.

### Q10: "What does your integration test actually prove?"

> It boots real Redis and RabbitMQ in containers — not mocks, not embeddings — points Spring at them via dynamic properties, publishes 200 transactions, and then asserts the full chain: all 200 consumed and scored (polling with Awaitility, since the pipeline is async); probabilities in [0,1] with positive latencies, proving the real ONNX model ran; Redis holding exactly cnt=200 for the card, proving the feature store folded in every transaction once; non-zero window counts on late transactions, proving sliding-window reads; and percentile keys present in the stats snapshot. One test, but it exercises messaging, features, inference, and metrics together — the failure modes that unit tests with mocks structurally cannot catch.

### Q11: "What happens if a consumer crashes mid-message? What are your delivery semantics?"

> The queue is durable and messages are unacked until the listener method returns; a crash requeues the message, so it's at-least-once delivery. The consequence is possible reprocessing, and my feature store isn't idempotent — a redelivered transaction would increment the counters twice. For the demo that's acceptable; the production fix is idempotency: track processed transaction ids (the ZSET members already are txn ids, so a `ZSCORE` existence check is nearly free), or make the whole update a Lua script keyed on txn id. I'd call that the most honest known-gap in the design.

### Q12: "How does backpressure work in your pipeline?"

> The queue is the buffer: if consumers fall behind, depth grows and the broker stops pushing beyond prefetch — 64 unacked per consumer — so consumers are never overwhelmed; latency degrades visibly (e2e includes queue wait) before anything breaks. On the fan-out side the WebSocket stream is sampled with a CAS-guarded rate limit, so slow browsers can't back up the pipeline. What's missing for production is producer-side flow control and a max queue depth policy — RabbitMQ supports both.

### Q13: "Why is `OrtSession` shared across threads, and why is that safe?"

> ONNX Runtime documents `OrtSession.run()` as thread-safe: the session holds immutable graph state and each run gets its own execution frame. So one session — one copy of the model in memory — serves all 4–8 listener threads concurrently. The per-call objects, the input tensor and the result, are not shared and are closed with try-with-resources because they wrap native off-heap memory that the GC doesn't manage.

### Q14: "Concurrency in `PipelineMetrics` — why the mix of atomics and synchronized?"

> Two different cost profiles. The per-transaction path runs 500 times a second, so it's atomics and a lock-free ring buffer — the write races are benign because a reservoir tolerates slot overwrites by design. The deques (throughput timestamps, recent transactions) are bounded and small, so short synchronized blocks are simpler to reason about and contention is immeasurable. Choosing lock-free everywhere would be complexity without payoff; choosing locks on the hot path would serialize consumers.

### Q15: "What is train/test leakage and where could this project have leaked?"

> Leakage is information from evaluation data reaching training. Three spots I guarded: SMOTE after the split, never before, so no synthetic reflections of test frauds enter training; rolling features computed strictly from *past* transactions in time order, so no future information leaks into a feature; and the replay stream is exclusively the held-out 20% — the live demo never shows the model data it trained on, which is also what makes the live-recall number legitimate.

### Q16: "Where does this design break at 100× the load?"

> At 50K tps: single Redis becomes the bottleneck first — the fix is hashing cards across a Redis Cluster, which the key design (`fs:{cardId}:*`) already supports since all of a card's keys share the hash tag. The single queue would move to a partitioned setup (or Kafka with card-id keys) to preserve per-card ordering across many consumers. The in-memory simple STOMP broker would become a real broker relay. Metrics would move to proper histograms (HdrHistogram) with export. The scoring stage itself scales horizontally already — it's stateless; all state lives in Redis and the broker.

---

## Potential Improvements

### Reliability

1. **Idempotent feature updates** — dedupe on transaction id (ZSCORE existence check or Lua script) so at-least-once redelivery can't double-count.
2. **Dead-letter queue** — poison messages currently retry; a DLX with a retry limit is the standard fix.
3. **Publisher confirms** — the simulator fire-and-forgets; confirms + a retry buffer would give a real delivery guarantee story.
4. **Graceful degradation** — if Redis is down, score on model features alone (V1–V28 + amount) with a "degraded" flag instead of failing the message.

### ML

5. **Threshold tuning** — pick the operating point from the PR curve against an explicit cost matrix instead of 0.5.
6. **Probability calibration** — tree ensembles are miscalibrated; Platt scaling / isotonic regression would make "0.9" mean 90%.
7. **Drift monitoring** — population-stability index on feature distributions; alert when serving drifts from training.
8. **Shadow deployment** — score with a candidate model alongside the champion, compare online before switching.
9. **Richer features** — merchant risk scores, geo-velocity, device fingerprints; the feature-store pattern extends naturally.

### Performance

10. **Micro-batched inference** — collect up to N transactions per few ms and run one [N,32] tensor; ONNX amortizes per-call overhead well.
11. **Redis Cluster with hash tags** — `fs:{cardId}` already co-locates each card's keys for clean sharding.
12. **HdrHistogram + Micrometer export** — proper histograms shipped to Prometheus/Grafana instead of hand-rolled reservoirs.

### Testing & Ops

13. **Load test as a test** — assert p99 < X ms at Y tps in CI using the simulator against Testcontainers infra.
14. **Contract test for the feature vector** — a golden-file test pinning the 32-feature layout between train.py and FraudScorer.
15. **GitHub Actions CI** — build, Spotless check, ruff/ty, integration tests (Testcontainers runs fine in CI).
16. **Containerize the app itself** — currently only infra is in Compose; a multi-stage Dockerfile completes the story.

---

## Quick Revision Cheatsheet

### Project in 3 Sentences

PulseGuard scores credit-card transactions for fraud in real time: RabbitMQ streams them to concurrent Spring consumers that pull rolling behavioral features from a Redis feature store (one pipelined round trip) and run an XGBoost model in-process via ONNX Runtime (~70 µs). The model is trained in Python with SMOTE on the ULB fraud dataset, and the training script computes identical features with identical read-before-write semantics — offline/online parity is the core design idea. Ground-truth labels ride along with replayed traffic, so the pipeline reports live recall as it runs; Testcontainers proves the whole chain against real Redis and RabbitMQ.

### Key Files

| File | One-Liner |
|---|---|
| `training/train.py` | download → parity features → SMOTE → XGBoost → ONNX (ZipMap stripped, parity-checked) |
| `features/FeatureStore.java` | ZSET window + HASH aggregates, 9 commands in 1 pipelined trip, read-before-write |
| `scoring/FraudScorer.java` | float[1][32] tensor → shared OrtSession → P(fraud) |
| `pipeline/TransactionConsumer.java` | @RabbitListener(4–8): features → inference → metrics → STOMP fan-out |
| `pipeline/TransactionSimulator.java` | replays 56,962 held-out txns at N tps, absolute-deadline pacing |
| `metrics/PipelineMetrics.java` | atomics + 4096 ring reservoirs → p50/p95/p99, 10 s throughput window |
| `config/RabbitConfig.java` | exchange `pulseguard.tx` → queue `pulseguard.transactions`, JSON converter |
| `config/OnnxConfig.java` | one OrtSession bean, closed by the container |
| `PipelineIntegrationTest.java` | Testcontainers: 200 txns through the full chain, asserts scores + Redis + percentiles |

### The 32-Feature Contract

```
[Amount, V1..V28, txn_count_60s, time_since_last(≤3600), amount_ratio]
```

Same order in train.py and FraudScorer. Guarded by the export-time parity check.

### Redis Keys (per card)

```
fs:{card}:w  ZSET  txnId → tsMillis   sliding 60s window (ZREMRANGEBYSCORE + ZCARD)
fs:{card}:s  HASH  cnt / sum / last   lifetime aggregates (HINCRBY / HINCRBYFLOAT / HSET)
both: EXPIRE 2h; reads before writes in one pipeline
```

### Key Numbers

| Fact | Value |
|---|---|
| Dataset | 284,807 txns / 492 frauds (0.172%, ~577:1) |
| Split | 80/20 stratified; SMOTE on train only: 227,845 → 454,902 rows |
| Model | XGBoost, 400 trees, depth 6, lr 0.1, hist |
| ROC-AUC / PR-AUC | 0.978 / 0.883 |
| Recall / Precision @ 0.5 | 0.867 / 0.810 (98 test frauds) |
| ONNX model size / parity diff | 795 KB / 0.000000 |
| Replay stream | 56,962 held-out txns |
| Features | 32 (Amount + V1–28 + 3 behavioral) |
| Sustained load | 500 txn/s |
| E2E latency | p50 2.5 ms / p95 4.2 ms / p99 6.8 ms |
| Inference | p50 69 µs |
| Feature store round trip | p50 1.7 ms |
| Consumers / prefetch | 4–8 / 64 |
| Reservoir / throughput window | 4096 samples / 10 s |
| Redis commands per txn / round trips | 9 / 1 |
| Stack versions | Java 25, Spring Boot 4.0.3, Testcontainers 2.0.3 |

### Run Commands

```bash
cd training && uv sync && uv run python train.py && cd ..   # train (writes model/ + data/)
docker compose up -d                                        # Redis :6379, RabbitMQ :5672/:15672
mvn spring-boot:run                                         # app + dashboard on :8080
mvn test                                                    # Testcontainers integration test
curl -X POST 'localhost:8080/api/simulator/start?tps=500'   # crank the load
curl localhost:8080/api/stats                               # counters + percentiles
```
