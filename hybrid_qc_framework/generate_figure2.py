import matplotlib.pyplot as plt
import numpy as np

# Data generation
categories = ['Classical', 'Quantum', 'Hybrid-QC']

# 1. Latency Reduction (ms) - lower is better
latency_data = [150.0, 10.5, 12.0]

# 2. Efficiency Improvement (Tasks/sec) - higher is better
efficiency_data = [50, 450, 480]

# 3. Resource Utilization (%) - lower is better
resource_data = [95, 99, 65]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 2: Performance Comparison Results', fontsize=16, fontweight='bold')

# Plot 1: Latency Reduction
colors = ['#ff9999','#66b3ff','#99ff99']
ax1.bar(categories, latency_data, color=colors)
ax1.set_title('Latency (ms)\nLower is Better')
ax1.set_ylabel('Milliseconds (ms)')
for i, v in enumerate(latency_data):
    ax1.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

# Plot 2: Efficiency Improvement
ax2.bar(categories, efficiency_data, color=colors)
ax2.set_title('Efficiency (Tasks/sec)\nHigher is Better')
ax2.set_ylabel('Tasks / second')
for i, v in enumerate(efficiency_data):
    ax2.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# Plot 3: Resource Utilization
ax3.bar(categories, resource_data, color=colors)
ax3.set_title('Resource Utilization (%)\nLower is Better (optimal blend)')
ax3.set_ylabel('Utilization (%)')
ax3.set_ylim(0, 110)
for i, v in enumerate(resource_data):
    ax3.text(i, v + 2, f"{v}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_performance_comparison.png', dpi=300)
print("Saved figure to figure2_performance_comparison.png")
