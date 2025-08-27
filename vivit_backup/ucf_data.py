import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ----------------------- Configuration -----------------------
# Paths to the UCF101 data and file lists – update these paths as needed.
data_root = "./UCF101/UCF-101"
class_index_file = "./ucfTrainTestlist/classInd.txt"
trainlist_files = [
    "./ucfTrainTestlist/trainlist01.txt",
    "./ucfTrainTestlist/trainlist02.txt",
    "./ucfTrainTestlist/trainlist03.txt"
]
testlist_files = [
    "./ucfTrainTestlist/testlist01.txt",
    "./ucfTrainTestlist/testlist02.txt",
    "./ucfTrainTestlist/testlist03.txt"
]

# ----------------------- Load Class Mapping -----------------------
# Read classInd.txt to map class indices to class names.
class_indices = {}
with open(class_index_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            idx, class_name = parts
            class_indices[int(idx)] = class_name

# Create ordered list of class names (zero-indexed)
data_subset = [class_indices[i] for i in sorted(class_indices.keys())]
print("Classes:", data_subset)

# ----------------------- Process Training Files -----------------------
train_videos = []
train_labels = []

for trainlist in trainlist_files:
    with open(trainlist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected format: "relative_video_path label"
            try:
                video_rel_path, label_str = line.split()
            except ValueError:
                continue  # skip malformed lines
            label = int(label_str) - 1  # Convert to zero-indexed label
            video_path = os.path.join(data_root, video_rel_path)
            # Optionally check for file existence:
            if not os.path.isfile(video_path):
                print(f"Warning: {video_path} not found!")
                continue
            train_videos.append(video_path)
            train_labels.append(label)

print(f"Loaded {len(train_videos)} training videos.")

# Create a DataFrame for the training set.
train_df = pd.DataFrame({"video_path": train_videos, "label": train_labels})
train_df["class_name"] = train_df["label"].apply(lambda x: data_subset[x])

# ----------------------- Process Test Files -----------------------
test_videos = []
test_labels = []

for testlist in testlist_files:
    with open(testlist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Some test lists might include labels; if not, infer from the directory name.
            if len(parts) == 2:
                video_rel_path, label_str = parts
                label = int(label_str) - 1  # zero-index conversion
            else:
                video_rel_path = parts[0]
                class_name = video_rel_path.split('/')[0]
                try:
                    label = data_subset.index(class_name)
                except ValueError:
                    print(f"Warning: Class {class_name} not found in class list!")
                    continue
            video_path = os.path.join(data_root, video_rel_path)
            if not os.path.isfile(video_path):
                print(f"Warning: {video_path} not found!")
                continue
            test_videos.append(video_path)
            test_labels.append(label)

print(f"Loaded {len(test_videos)} test videos.")

# Create a DataFrame for the test set.
test_df = pd.DataFrame({"video_path": test_videos, "label": test_labels})
test_df["class_name"] = test_df["label"].apply(lambda x: data_subset[x])

# ----------------------- Split Test Set into Validation and Test -----------------------
# For analysis, split the test set evenly into validation and final test subsets.
val_df, final_test_df = train_test_split(
    test_df,
    test_size=0.5,
    random_state=42,
    stratify=test_df["label"]
)
print(f"Validation samples: {len(val_df)}, Test samples: {len(final_test_df)}")

# ----------------------- Data Analysis & Visualization -----------------------
def plot_combined_distribution(train_df, val_df, test_df, filename):
    # Combine dataframes with a new column indicating the split
    train_df['split'] = 'Train'
    val_df['split'] = 'Validation'
    test_df['split'] = 'Test'
    combined_df = pd.concat([train_df, val_df, test_df])

    # Group by class and split to count occurrences
    distribution = combined_df.groupby(['class_name', 'split']).size().reset_index(name='count')

    # Plot the combined distribution
    plt.figure(figsize=(16, 8))
    sns.barplot(data=distribution, x='class_name', y='count', hue='split', 
                palette={'Train': 'blue', 'Validation': 'green', 'Test': 'red'})
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of Videos")
    plt.title("Class Distribution Across Splits")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined plot: {filename}")

# Plot combined distributions for training, validation, and test splits.
plot_combined_distribution(train_df, val_df, final_test_df, "combined_distribution.png")

# ----------------------- Save Splits to CSV -----------------------
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
final_test_df.to_csv("test_split.csv", index=False)
print("Data splits saved to CSV files.")

# ----------------------- Summary Statistics -----------------------
def print_summary(df, split_name):
    total = len(df)
    class_counts = df['class_name'].value_counts().sort_index()
    print(f"\n{split_name} Summary:")
    print(f"Total videos: {total}")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

print_summary(train_df, "Training Set")
print_summary(val_df, "Validation Set")
print_summary(final_test_df, "Test Set")
