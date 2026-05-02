import os

def calculate_weights_from_folders(base_path):
    # Standard RAF-DB labels: 
    # 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 
    # 5: Sadness, 6: Anger, 7: Neutral
    counts = {}
    
    # Iterate through folders 1-7
    for i in range(1, 8):
        folder_name = str(i)
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            counts[i] = file_count
        else:
            print(f"Warning: Folder {folder_name} not found at {folder_path}")

    if not counts:
        print("No data found. Check your path!")
        return None

    print(f"Class Counts: {counts}")
    
    # Find the smallest class count (N_min)
    n_min = min(counts.values())
    
    # Calculate weights: W = N_min / N_target
    weights = {cls: n_min / count for cls, count in counts.items()}
    
    return weights

# UPDATE THIS PATH to your local 'train' folder
DATA_PATH = r"C:\Users\chinn\OneDrive\Desktop\Datasets\RAF-DB\DATASET\train"

if __name__ == "__main__":
    weights = calculate_weights_from_folders(DATA_PATH)
    if weights:
        print("\n--- FINAL WEIGHTS FOR YOUR LOSS FUNCTION ---")
        for cls in sorted(weights.keys()):
            print(f"Class {cls}: {weights[cls]:.4f}")