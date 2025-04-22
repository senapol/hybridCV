import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

# need to update to csv
methods = ['YOLO', 'BG', 'YOLO+LK', 'BG+LK', 'HYBRID']
detection_rates = [36.67, 6.67, 100.0, 43.33, 100.0]
max_consecutive_misses = [7, 21, 0, 11, 0]
avg_gap_lengths = [3.60, 3.50, 0.00, 6.00, 0.00]
processing_times = [200.25, 131.42, 7.94, 2.93, 3.54]

# define different colors for each method
colors = ['#ff3333', '#3366ff', '#ffcc00', '#cc33ff', '#ff9933']

# 1. Bar Chart Visualization
plt.figure(figsize=(14, 10))

# Create 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Detection rates
axes[0, 0].bar(methods, detection_rates, color=colors)
axes[0, 0].set_title('Detection Rate (%)', fontsize=14)
axes[0, 0].set_ylim(0, 105)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(detection_rates):
    axes[0, 0].text(i, v+2, f"{v}%", ha='center', fontweight='bold')

# Max consecutive misses
axes[0, 1].bar(methods, max_consecutive_misses, color=colors)
axes[0, 1].set_title('Max Consecutive Frames Without Detection', fontsize=14)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(max_consecutive_misses):
    axes[0, 1].text(i, v+0.5, str(v), ha='center', fontweight='bold')

# Average gap length
axes[1, 0].bar(methods, avg_gap_lengths, color=colors)
axes[1, 0].set_title('Average Gap Length (frames)', fontsize=14)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(avg_gap_lengths):
    axes[1, 0].text(i, v+0.2, str(v), ha='center', fontweight='bold')

# Processing time
axes[1, 1].bar(methods, processing_times, color=colors)
axes[1, 1].set_title('Average Processing Time (ms/frame)', fontsize=14)
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(processing_times):
    axes[1, 1].text(i, v+5, f"{v} ms", ha='center', fontweight='bold')

# Formatting for all subplots
for ax in axes.flat:
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)

plt.suptitle('Tennis Ball Detection Method Comparison', fontsize=16, y=0.98)
plt.figtext(0.5, 0.01, 'Frames 1020 to 1050', ha='center', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('comparison_metrics.png', dpi=300)

# 2. Radar Chart for Multi-dimensional Comparison
plt.figure(figsize=(10, 8))

# Normalize each metric to a 0-1 scale for the radar chart
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [1.0 if val > 0 else 0.0 for val in values]
    return [(val - min_val) / (max_val - min_val) for val in values]

# For metrics where higher is better
norm_detection_rates = normalize(detection_rates)

# For metrics where lower is better (invert the normalization)
norm_max_misses = [1 - x for x in normalize(max_consecutive_misses)]
norm_avg_gaps = [1 - x for x in normalize(avg_gap_lengths)]
norm_proc_times = [1 - x for x in normalize(processing_times)]

# Combine all metrics
metrics = ['Detection Rate', 'Processing Speed', 'Consecutive Detection', 'Gap Recovery']
num_metrics = len(metrics)

angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Plot each method
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

for i, method in enumerate(methods):
    values = [norm_detection_rates[i], norm_proc_times[i], norm_max_misses[i], norm_avg_gaps[i]]
    values += values[:1]  # Close the loop
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=method)
    ax.fill(angles, values, color=colors[i], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_yticks([])
ax.set_title('Method Comparison - All Metrics', fontsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('radar_comparison.png', dpi=300)

# 3. Formatted Table Visualization
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table_data = [
    [f"{val:.2f}%" for val in detection_rates],
    [str(val) for val in max_consecutive_misses],
    [f"{val:.2f}" for val in avg_gap_lengths],
    [f"{val:.2f} ms" for val in processing_times]
]

row_labels = [
    'Detection Rate',
    'Max Consecutive Misses',
    'Avg Gap Length (frames)',
    'Processing Time'
]

table = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=methods,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

# Color the column headers according to the method colors
for i, color in enumerate(colors):
    table[(0, i)].set_facecolor(color)
    table[(0, i)].set_text_props(color='white', fontweight='bold')

plt.title('Detection Method Metrics Summary', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('metrics_table.png', dpi=300)

# 4. Create a cost-benefit scatter plot
plt.figure(figsize=(10, 6))

# Plot detection rate vs processing time
plt.scatter(processing_times, detection_rates, s=200, c=colors, alpha=0.8)

# Add method labels to each point and store them in a list for adjustment
texts = []
for i, method in enumerate(methods):
    if method == 'HYBRID':
        xytext_offset = (-30,-20)
    elif method == 'YOLO+LK':
        xytext_offset = (10,5)
    else:
        xytext_offset = (5, 5)
    texts.append(plt.annotate(method, 
                               (processing_times[i], detection_rates[i]),
                               xytext=xytext_offset,
                               textcoords='offset points',
                               fontsize=12,
                               fontweight='bold'))


plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Processing Time (ms/frame)', fontsize=14)
plt.ylabel('Detection Rate (%)', fontsize=14)
plt.title('Detection Rate vs Processing Time', fontsize=16)
plt.tight_layout()
plt.savefig('rate_vs_time.png', dpi=300)