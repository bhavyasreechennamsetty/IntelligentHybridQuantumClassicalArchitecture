# Hybrid Quantum-Classical Framework - Detailed Technical Documentation

This document provides a comprehensive, deep-dive technical analysis of the Hybrid Quantum-Classical Framework. This system is designed to prototype a three-layer architecture (Preprocessing, Quantum Acceleration, Orchestration) for processing high-velocity data streams using NISQ (Noisy Intermediate-Scale Quantum) devices.

Below is a detailed breakdown of every file in the project, explaining its architectural role, internal logic, algorithms, and design decisions.

---

## 1. System Entry Point

### `hybrid_qc_framework/main.py`

**Architectural Role:**
The `main.py` file serves as the **orchestrator and entry point** for the entire application. It is responsible for bridging the asynchronous data ingestion layer with the synchronous processing pipeline. It acts as the "glue" that instantiates all modular components (Ingestion, Preprocessing, Quantum, and Decision Engine) and manages the runtime lifecycle of the application.

**Detailed Logic & Workflow:**
1.  **Initialization Phase:**
    -   The script begins by setting up imports to ensure local modules are resolvable.
    -   It instantiates the core components: `StreamSimulator` (Data Source), `NoiseFilter`, `FeatureEngineer`, `DimensionalityReducer` (Layer 1), `QuantumExecutor` (Layer 2), and `DecisionEngine` (Layer 3).
    -   A global, thread-safe `queue.Queue` is created to act as the buffer between the data generation thread and the main processing loop.

2.  **Asynchronous Ingestion:**
    -   A dedicated thread, `ingestion_worker`, is spawned. This function calls `stream_simulator.stream()` and pushes packets into the shared queue.
    -   **Design Decision**: Threading is used here to decouple data generation from processing. In a real-world scenario, data arrives independently of the system's processing speed. If the main loop sends a job to a real QPU (waiting in a queue), the ingestion must not block. The queue allows for buffering and backpressure handling.

3.  **The Main Event Loop:**
    -   The `while True` loop represents the continuous processing capability.
    -   **Layer 1 (Preprocessing)**: Raw data is fetched from the queue. It passes through the `NoiseFilter` and `FeatureEngineer`. Crucially, the `DimensionalityReducer` performs *online learning* here using `partial_fit`, updating its internal model with every new batch of data before reducing the dimensions.
    -   **Layer 3 (Decision making)**: Before execution, the `Monitor` checks system health (latency). The `DecisionEngine` uses the RL agent to decide: *Should this packet go to the Quantum backend or stay Classical?*
    -   **Layer 2 (Execution)**: Based on the decision, the data is routed.
        -   **Quantum Path**: usage of `QuantumExecutor` to run a search/optimization circuit.
        -   **Classical Path**: A lightweight fallback calculation (e.g., statistical mean) is performed.
    -   **Feedback Loop**: Post-execution, the `DecisionEngine` calculates a reward based on latency and outcome, updating the RL agent. It also adjusts global parameters (ingestion rate, target dimensions) to stabilize the system.

**Key Code Elements:**
-   `ingestion_worker(sim, q)`: The producer function running in a separate daemon thread.
-   `data_queue`: The synchronization primitive handling backpressure (simulated via queue size).
-   `tracker`: An instance of `PerformanceTracker` that aggregates latency statistics across the pipeline.

---

## 2. Ingestion Layer

### `hybrid_qc_framework/ingestion/stream_simulator.py`

**Architectural Role:**
This file mocks the external environment. In a production version, this would be replaced by a driver for a hardware sensor array, a Kafka consumer, or a network socket. Its role is to provide a controllable, infinite stream of high-dimensional numerical data.

**Detailed Logic:**
The simulator is designed to produce data that is "structured but noisy," mimicking real-world sensor signals (like LiDAR or RF streams) that might require quantum processing.
-   **Signal Generation**: It generates a base signal using a sine wave function: `np.sin(linespace + time)`. This creates a wave pattern that shifts over time, ensuring the data isn't static.
-   **Noise Injection**: Gaussian noise (`np.random.normal`) is added to this signal. This is critical for testing the `NoiseFilter` downstream; without noise, the filter's effect would be invisible.
-   **Rate Limiting**: To simulate a specific frequency (e.g., 10 Hz), the `stream()` generator calculates the elapsed time for each iteration and sleeps for the remainder of the interval.

**Code Structure:**
-   `StreamSimulator` Class:
    -   `__init__`: Accepts `data_rate_hz` (frequency) and `n_features` (dimensionality, default 50).
    -   `stream()`: A Python generator yielding dictionaries. This pattern allows the consumer to iterate over data without loading it all into memory, a requirement for streaming systems.
    -   `stop()`: A thread-safe flag (`self.running`) to cleanly terminate the infinite loop.

---

## 3. Preprocessing Layer

### `hybrid_qc_framework/preprocessing/noise_filter.py`

**Architectural Role:**
The first line of defense in the classical preprocessing layer. Its goal is to improve data quality *before* expensive quantum encoding takes place. Quantum states are fragile; feeding noisy garbage into a quantum circuit yields noisy garbage results ("GIGO").

**Detailed Logic:**
-   The class wraps `scipy.ndimage.gaussian_filter1d`.
-   **Algorithm**: Gaussian smoothing involves convolving the input signal with a Gaussian function window. This effectively averages each data point with its neighbors, weighted by distance. High-frequency jitters (noise) are smoothed out, while the improved lower-frequency underlying trend (the signal) is preserved.
-   **Parameters**: The `sigma` parameter acts as a knob. A higher sigma means more smoothing (blurring), while a lower sigma preserves more detail but also more noise.

### `hybrid_qc_framework/preprocessing/feature_engineering.py`

**Architectural Role:**
Responsible for transforming raw physical values into a format suitable for machine learning or quantum embedding. In this prototype, it focuses on **Scaling/Normalization**.

**Detailed Logic:**
-   **Normalization**: Quantum Angle Encoding (used later) typically maps values to rotational angles, often in the range $[0, 2\pi]$ or $[-1, 1]$. If raw sensor data has values like 500.0 or -200.0, passing them directly to a rotation gate `Ry(theta)` results in rapid oscillation (since $500\pi$ is just a lot of full rotations), making the encoding sensitive to minute errors.
-   The `extract_features` method performs Min-Max scaling to map the input vector to the $[0, 1]$ range.
-   **Future Extensibility**: The class is structured to allow adding stats extraction (mean, kurtosis) which could be appended to the vector as "meta-features" for the decision engine.

### `hybrid_qc_framework/preprocessing/dimensionality_reduction.py`

**Architectural Role:**
This is the most critical classical component. Current quantum computers (NISQ) have very few reliable qubits (e.g., 5 to 100). Real-world data often has hundreds of features. This module compresses the input space (e.g., 50 features) down to the available qubit count (e.g., 4) while preserving the maximum variance/information.

**Detailed Logic (Incremental PCA):**
-   Standard PCA (Principal Component Analysis) requires the entire dataset to compute the covariance matrix. This is impossible in a streaming context where data implies infinity.
-   **Algorithm**: This file implements `sklearn.decomposition.IncrementalPCA`.
    -   **Buffering**: It maintains a `buffer` list. It does not fit the model on every single packet (which would be inefficient). Instead, it accumulates a "mini-batch" (e.g., 10 packets).
    -   **Partial Fit**: Once the buffer acts, `ipca.partial_fit(batch)` is called. This updates the singular value decomposition (SVD) of the data incrementally.
    -   **Projection**: The `reduce()` method projects the high-dim vector onto the principal components learned *so far*.
-   **Dynamic Resizing**: The `update_target_dim` method allows the Orchestrator to fundamentally change the model structure at runtime (e.g., dropping from 8 qubits to 4 qubits to save time), triggering a reset of the internal learner.

---

## 4. Quantum Layer

### `hybrid_qc_framework/quantum/circuits.py`

**Architectural Role:**
A library of quantum circuit templates. It abstracts the Qiskit circuit construction logic away from the execution logic.

**Detailed Logic:**
1.  **`feature_map_circuit` (Data Encoding):**
    -   How do we get classical numbers into a quantum state? This method implements **Angle Encoding**.
    -   It creates a `QuantumCircuit` with $N$ qubits.
    -   For each data point $x_i$ in the reduced vector, it applies a rotation $R_y(\pi \cdot x_i)$ to the $i$-th qubit. This maps the value $[0, 1]$ to a quantum state superposition on the Bloch sphere.
    -   **Design Decision**: Angle encoding is efficient ($O(1)$ depth) compared to Amplitude encoding ($O(2^N)$), making it ideal for streaming where latency is key.
2.  **`simple_search_oracle` (The Task):**
    -   This represents the "algorithm" we want to run (e.g., Grover's Search).
    -   To keep the simulation fast, it implements a mock oracle using a layer of Hadamard gates (superposition) followed by entanglement (CZ gates). This creates a complex, entangled state that is hard to simulate classically, serving as a proxy for "quantum advantage."

### `hybrid_qc_framework/quantum/executor.py`

**Architectural Role:**
The interface to the quantum hardware (or simulator). It handles the compilation (transpilation), job submission, result retrieval, and error mitigation.

**Detailed Logic:**
-   **Backend Selection**: It robustly checks for `qiskit_aer` (local high-performance C++ simulator). If missing, it falls back to Qiskit's reference `Sampler`. This ensures the code runs on any machine.
-   **Execution Flow**:
    1.  Constructs the circuit by combining the feature map and the oracle.
    2.  Adds measurement gates (`measure_all`).
    3.  Transpiles/Runs the circuit with a specified number of "shots" (e.g., 1024 repetitions) to build a probability distribution.
-   **Naive Error Mitigation**:
    -   NISQ devices are noisy. A bitstring that appears only once in 1024 shots is likely just noise.
    -   The code filters out counts that fall below a 5% threshold (`total_shots * 0.05`). This acts as a simple "high-pass filter" for probability, returning only statistically significant results.
-   **Result Parsing**: It identifies the "most frequent bitstring" as the answer to the optimization problem.

---

## 5. Orchestration Layer (The "Brain")

### `hybrid_qc_framework/orchestration/decision_engine.py`

**Architectural Role:**
This file implements the **Layer 3 Feedback Loop**. It connects system metrics (latency) to system controls (routing and parameter tuning). It is the realization of the "Hybrid" concept—intelligently choosing between CPU and QPU.

**Detailed Logic:**
-   **`Monitor` Class**: A simple wrapper that interprets raw lists of latency numbers into a semantic status (`"high_latency"`, `"normal"`, `"low_latency"`).
-   **`DecisionEngine` Class**:
    1.  **Routing (`decide_execution_path`)**:
        -   It constructs a "State" tuple based on latency and data complexity (standard deviation).
        -   It queries the `QLearningAgent` for an action (0=Classical, 1=Quantum).
        -   **Reward Function**: This is the core logic. It rewards the agent for keeping latency low. It *specifically* adds a bonus reward if the agent chooses Quantum for "high complexity" data, encoding the business logic that "Quantum is valuable for complex problems."
        -   It passes the reward back to the agent to update the Q-table (learning step).
    2.  **Parameter Tuning (`adjust_parameters`)**:
        -   Independent of the per-packet routing, this adjusts the *environment*.
        -   If `high_latency`: It lowers the Ingestion Rate and reduces the Target Dimension (simulating a "Panic Mode" to recover stability).
        -   If `low_latency`: It safely increases the rate and dimension, probing for higher throughput and better precision.

### `hybrid_qc_framework/orchestration/rl_agent.py`

**Architectural Role:**
A generic implementation of a Reinforcement Learning agent (Q-Learning). It is decoupled from the specific domain logic, handling only states, actions, and Q-values.

**Detailed Logic:**
-   **Q-Learning**: A model-free RL algorithm. It learns the "Quality" (Q-value) of taking a specific Action in a specific State.
-   **State Representation**: The `get_state_key` function discretizes continuous values (latency 0.045s -> "L_LOW") into discrete buckets. This is necessary for tabular Q-learning, otherwise the state space would be infinite.
-   **Q-Table**: A dictionary mapping `(State)` to `[Q_action_0, Q_action_1]`.
-   **Epsilon-Greedy Strategy**: In `choose_action`, the agent mostly chooses the best known action (Exploitation), but with probability `epsilon` (e.g., 10%), it picks a random action (Exploration). This ensures it doesn't get stuck in a suboptimal loop and discovers new strategies.
-   **Bellman Equation**: The `learn` method updates the Q-values using the standard Bellman update rule:
    $Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s',a') - Q(s,a)]$

---

## 6. Metrics & Utilities

### `hybrid_qc_framework/metrics/performance.py`

**Architectural Role:**
Telephmetry and state tracking. Ensuring all components have a shared source of truth regarding system performance.

**Detailed Logic:**
-   **`SystemMetrics` Dataclass**: A clean data structure to hold the snapshot of the system (counts, current latency, current parameters).
-   **`PerformanceTracker` Class**:
    -   **Rolling Window**: To calculate "current latency," it doesn't average *all* history (which would react too slowly). It keeps a list of the last 50 latencies. This makes the `DecisionEngine` responsive to recent spikes.
    -   It provides thread-safe counters for quantum vs. classical executions, essential for the final validation report.
