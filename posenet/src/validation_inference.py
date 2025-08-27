import os
import subprocess

# Paths
SOURCE_FOLDER = "data/source_images3"
OUTPUT_FOLDER = "output/valid"
MODEL_PATH = "model/trained_classifier.pickle"
SCRIPT_PATH = "src/s5_test.py"  # Ensure this points to the correct location

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all action_timestamp folders in source_images3
action_timestamps = [d for d in os.listdir(SOURCE_FOLDER) if os.path.isdir(os.path.join(SOURCE_FOLDER, d))]

for action_timestamp in action_timestamps:
    input_path = os.path.join(SOURCE_FOLDER, action_timestamp)
    output_path = os.path.join(OUTPUT_FOLDER, action_timestamp)

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    print(f"Running model on {action_timestamp}...")
    
    # Run the s5_test.py script with the required arguments
    subprocess.run([
        "python", SCRIPT_PATH,
        "--model_path", MODEL_PATH,
        "--data_type", "folder",
        "--data_path", input_path,
        "--output_folder", OUTPUT_FOLDER
    ])

print("Processing complete. Outputs saved in:", OUTPUT_FOLDER)
