import os

# MODIFY THIS PATH TO YOUR DATASET DIRECTORY
DATASET_DIRECTORY = r"C:/ForKrishna/VideoAnalytics/datasets/ucf_crime"  # Replace with your actual dataset path

def generate_classes_file():
    """
    Generate truncated_classes.txt from folder names in the dataset directory
    """
    
    # Check if directory exists
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"Error: Directory '{DATASET_DIRECTORY}' does not exist!")
        print("Please modify the DATASET_DIRECTORY variable in this script.")
        return
    
    try:
        # Get all folders (class names)
        all_items = os.listdir(DATASET_DIRECTORY)
        class_folders = [item for item in all_items 
                        if os.path.isdir(os.path.join(DATASET_DIRECTORY, item))]
        
        # Sort alphabetically
        class_folders.sort()
        
        if not class_folders:
            print(f"No folders found in '{DATASET_DIRECTORY}'")
            return
        
        # Write to truncated_classes.txt in the same directory as this script
        output_file = os.path.join(os.path.dirname(__file__), "truncated_classes.txt")
        
        with open(output_file, 'w') as f:
            for class_name in class_folders:
                f.write(f"{class_name}\n")
        
        print(f"✅ Successfully generated truncated_classes.txt with {len(class_folders)} classes")
        print(f"📁 Output file: {output_file}")
        print(f"📂 Source directory: {DATASET_DIRECTORY}")
        print("\n📋 Classes found:")
        for i, class_name in enumerate(class_folders, 1):
            print(f"  {i:2d}. {class_name}")
        
        # Show file content preview
        print(f"\n📄 Content written to truncated_classes.txt:")
        print("-" * 40)
        with open(output_file, 'r') as f:
            content = f.read()
            print(content)
        print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔍 Generating class names from dataset folders...")
    print(f"📂 Looking in: {DATASET_DIRECTORY}")
    print()
    
    generate_classes_file()