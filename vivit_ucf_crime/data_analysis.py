import os
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Paths
dataset_path = "./kinetics400_5per/train"
class_file = "truncated_classes.txt"
output_image = "dataset_distribution.png"

# Read the file and organize categories
categories = defaultdict(list)
current_category = None
category_order = []  # Maintain order of categories

with open(class_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("//"):
            if line.startswith("//"):
                current_category = line.strip("//").strip()
                if current_category not in category_order:
                    category_order.append(current_category)
        else:
            categories[current_category].append(line)

# Count videos per class
class_counts = {}
for category, classes in categories.items():
    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            class_counts[class_name] = len([f for f in os.listdir(class_folder) if f.endswith(".mp4")])
        else:
            class_counts[class_name] = 0  # If missing, assume 0 videos

# Sort classes by count
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
class_names = [c[0] for c in sorted_classes]
video_counts = [c[1] for c in sorted_classes]

# Map class names to categories
class_to_category = {cls: cat for cat, cls_list in categories.items() for cls in cls_list}

# Generate random colors for each category
category_colors = {cat: [random.random(), random.random(), random.random()] for cat in category_order}

# Total number of videos
total_videos = sum(video_counts)

# Figure setup
fig, ax = plt.subplots(figsize=(12, len(class_names) * 0.2))
category_bar_colors = [category_colors[class_to_category[cls]] for cls in class_names]

# Create bars with category-specific colors
bars = ax.barh(class_names, video_counts, color=category_bar_colors, edgecolor="black", height=0.8)
ax.set_xlabel("Number of Videos", fontsize=12)
ax.set_title(f"Dataset Distribution by Class (Total Videos: {total_videos})", fontsize=14, fontweight="bold")

# Draw category brackets (optional, for visual structure)
y_offset = 0.5  # Bracket vertical padding
category_positions = {}  # Track y-range of categories

for i, class_name in enumerate(class_names):
    category = class_to_category.get(class_name, "Uncategorized")
    if category not in category_positions:
        category_positions[category] = [i, i]  # Start and end index
    else:
        category_positions[category][1] = i  # Update end index

# Annotate categories with brackets (optional)
for category, (start, end) in category_positions.items():
    bracket_color = category_colors[category]
    ax.plot([-5, -5], [start - y_offset, end + y_offset], color=bracket_color, lw=2)  # Vertical line
    ax.plot([-5, 0], [start - y_offset, start - y_offset], color=bracket_color, lw=2)  # Top horizontal
    ax.plot([-5, 0], [end + y_offset, end + y_offset], color=bracket_color, lw=2)  # Bottom horizontal

# Customizing labels and aesthetics
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, fontsize=10)
ax.invert_yaxis()  # Highest counts at the top
plt.xlim(-10, max(video_counts) + 10)  # Adjust for category text

# Style the figure
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add a legend to map categories to colors
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in category_colors.values()]
labels = list(category_colors.keys())
ax.legend(handles, labels, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Tight layout for better space management
plt.tight_layout()

# Save and show
plt.savefig(output_image, dpi=300)
plt.show()

print(f"Total number of videos: {total_videos}")
print(f"Dataset distribution image saved as {output_image}")
