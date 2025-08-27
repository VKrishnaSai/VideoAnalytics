# Step 1 will be to get the data from source_images3/{action}_{timestamp}
# Our data for validation will be saved in valid/{action}_{timestamp}
# Now run the model on the data which is source_images3/{action_timestamp}/ and save outputs to output/valid/{action}_{timestamp}/
# we have to change output format from output/{timestamp}/ to output/valid/{action}_{timestamp}/
# model output will be saved in output/valid/{action}_{timestamp}/ and will be a .txt for each image

# Step 2, now, suppose we are validating an action in valid/{action}_{timestamp}/
# take note of the action and timestamp, access the related output folder which is output/valid/{action}_{timestamp}/
# now just check if {action} is present in output file, if it is: (its not as simple as variable, we will store more depth data, like what is the incorrect_prediction as well)
#    if it is correct prediction, increment correct_prediction by 1
#    if there is no prediction, increment no_prediction by 1
#    if there is incorrect prediction, increment incorrect_prediction by 1

# Step 3, now, at the end we must have correct_prediction + no_prediction + incorrect_prediction = total folder size of valid/{action}_{timestamp}/
# do this for everything....


# Step 4, then, concatenate results for each action, then we will plot a confusion matrix between the predicted and ground truth

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import ast

if True:  # Include project path
    import sys
    ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
    sys.path.append(ROOT)
    import utils.lib_commons as lib_commons

# Paths
cfg_all = lib_commons.read_yaml("config/config.yaml")
CLASSES = np.array(cfg_all["classes"])
VALID_FOLDER = "./data/source_images3/"
OUTPUT_FOLDER = "./output/valid/data/source_images3/"

# Initialize result tracking
results = {action: {"correct": 0, "incorrect": 0, "no_prediction": 0, "total": 0, "misclassified": defaultdict(int)} for action in CLASSES}

def process_action(action_timestamp):
    action, timestamp = action_timestamp.rsplit("_", 1)
    valid_path = os.path.join(VALID_FOLDER, action_timestamp)
    output_path = os.path.join(OUTPUT_FOLDER, action_timestamp, "skeletons")
    
    if not os.path.exists(output_path):
        print(f"Skipping {action_timestamp}, no output found in {output_path}")
        return
    
    files = os.listdir(valid_path)
    total_files = len(files)
    
    for file in files:
        output_file = os.path.join(output_path, file.replace(".png", ".txt"))
        
        if not os.path.exists(output_file):
            results[action]["no_prediction"] += 1
            continue
        
        with open(output_file, "r") as f:
            predictions = ast.literal_eval(f.read())
        
        if not predictions or not predictions[0]:
            results[action]["no_prediction"] += 1
            continue
        
        first_prediction = predictions[0][0][1]
        
        if action == first_prediction:
            results[action]["correct"] += 1
        else:
            results[action]["incorrect"] += 1
            results[action]["misclassified"][first_prediction] += 1
    
    results[action]["total"] = total_files
    print(f"Processed {action_timestamp}: {results[action]}")

# Process all validation folders
for action_timestamp in os.listdir(VALID_FOLDER):
    process_action(action_timestamp)

# Generate confusion matrix
action_indices = {action: i for i, action in enumerate(CLASSES)}
conf_matrix = np.zeros((len(CLASSES), len(CLASSES)))

for action, data in results.items():
    i = action_indices[action]
    conf_matrix[i, i] = data["correct"]
    for misclassified_action, count in data["misclassified"].items():
        if misclassified_action in action_indices:
            j = action_indices[misclassified_action]
            conf_matrix[i, j] = count

# Normalize the confusion matrix
normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.show()

# Calculate precision, recall, and F1-score for each class
for action, index in action_indices.items():
    tp = conf_matrix[index, index]
    fp = conf_matrix[:, index].sum() - tp
    fn = conf_matrix[index, :].sum() - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    results[action].update({
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    })

# Save extended results
with open("extended_validation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Validation complete. Extended results saved.")

# Save results
with open("validation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Validation complete. Results saved.")
