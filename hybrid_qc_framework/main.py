import time
import sys
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.stream_simulator import StreamSimulator
from preprocessing.noise_filter import NoiseFilter
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.dimensionality_reduction import DimensionalityReducer
from quantum.executor import QuantumExecutor
from metrics.performance import PerformanceTracker
from orchestration.decision_engine import DecisionEngine, Monitor
import numpy as np

# Shared thread-safe queue for the Async Pipeline
data_queue = queue.Queue(maxsize=100) # Buffer to handle bursts

def ingestion_worker(sim, q):
    """
    Independent Data Ingestion Thread.
    Never blocked by processing.
    """
    print("[THREAD] Ingestion Worker Started")
    for packet in sim.stream():
        if not sim.running:
            break
        try:
            # Non-blocking put better? No, if full we want backpressure or drop.
            # Here we block slightly to simulate backpressure if system is choked
            q.put(packet, timeout=1.0)
        except queue.Full:
            print("[WARN] Buffer Full - Dropping Packet!")
            
    print("[THREAD] Ingestion Worker Stopped")

def main():
    print("="*60)
    print("HYBRID QUANTUM-CLASSICAL FRAMEWORK - PHASE 2 (OPTIMIZED)")
    print("Features: Async Pipeline, RL Orchestration, Incr. PCA")
    print("="*60)
    
    # 1. Initialize Layers
    initial_rate = 10.0
    stream_sim = StreamSimulator(data_rate_hz=initial_rate)
    
    noise_filter = NoiseFilter()
    feature_eng = FeatureEngineer()
    dim_reducer = DimensionalityReducer(target_dim=4)
    q_executor = QuantumExecutor()
    
    tracker = PerformanceTracker()
    monitor = Monitor(tracker)
    decision_engine = DecisionEngine()
    
    # 2. Start Ingestion Thread (Async Architecture)
    ingest_thread = threading.Thread(target=ingestion_worker, args=(stream_sim, data_queue))
    ingest_thread.daemon = True
    stream_sim.running = True
    ingest_thread.start()
    
    print("[SYSTEM] Modules Loaded. Pipeline Active.")
    print("-" * 60)
    print(f"{'TIME':<10} | {'DECISION':<12} | {'LATENCY (s)':<12} | {'RATE':<5} | {'RES'}")
    print("-" * 60)
    
    try:
        start_global = time.time()
        max_duration = 30 # seconds
        
        while True:
            t_now = time.time()
            if t_now - start_global > max_duration:
                print("\n[SYSTEM] Demo duration reached.")
                break
                
            # Consume from Queue
            try:
                packet = data_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            loop_start = time.time()
            
            # --- LAYER 1: CLASSICAL PREPROCESSING ---
            raw_data = packet['data']
            cleaned_data = noise_filter.apply(raw_data)
            features = feature_eng.extract_features(cleaned_data)
            
            # Online Learning for PCA
            dim_reducer.learn(features)
            reduced_data = dim_reducer.reduce(features)
            
            # --- LAYER 3: DECISION ENGINE (RL) ---
            # Get current health 
            health, metrics = monitor.check_health()
            
            decision = decision_engine.decide_execution_path(packet, health, metrics.total_latency)
            
            result_summary = ""
            
            if decision == "quantum":
                # --- LAYER 2: QUANTUM ACCELERATION ---
                q_result = q_executor.execute_task(reduced_data, task_type="search_optimization")
                tracker.increment_quantum()
                result_summary = f"Q-Bits: {q_result['top_result']}"
            else:
                # Classical
                tracker.increment_classical()
                result_summary = f"C-Val: {np.mean(reduced_data):.2f}"
            
            loop_end = time.time()
            latency = loop_end - loop_start
            tracker.log_latency(latency)
            
            # --- FEEDBACK LOOP ---
            new_rate, new_dim = decision_engine.adjust_parameters(health, stream_sim.data_rate_hz, dim_reducer.target_dim)
            
            # Apply adjustments
            if new_rate != stream_sim.data_rate_hz:
                stream_sim.data_rate_hz = new_rate
            if new_dim != dim_reducer.target_dim:
                dim_reducer.update_target_dim(new_dim)
            tracker.update_state(new_rate, new_dim)
            
            print(f"{time.strftime('%H:%M:%S'):<10} | {decision:<12} | {latency:.4f}       | {int(new_rate):<5} | {result_summary}")
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by user.")
    finally:
        stream_sim.stop()
        ingest_thread.join(timeout=2)
        final_metrics = tracker.get_summary()
        print("="*60)
        print("FINAL OPTIMIZED REPORT")
        print(f"Classical Tasks: {final_metrics.classical_count}")
        print(f"Quantum Tasks:   {final_metrics.quantum_count}")
        print(f"Avg Latency:     {final_metrics.total_latency:.4f} s")
        print("="*60)

if __name__ == "__main__":
    main()
