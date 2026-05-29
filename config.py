import os

IS_COLAB = os.path.exists('/content/drive')

if IS_COLAB:
    # Google Colab Paths
    BASE_PATH = "/content/data/Datasets/RAF-DB"
    SAVE_DIR = "/content/drive/MyDrive/FER_Phase3_Results"
else:
    
    BASE_PATH = "C:/Users/Documents/Datasets/RAF-DB" 
    SAVE_DIR = "./results"

# Common file paths derived from the base
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
VAL_CSV = os.path.join(BASE_PATH, "test_labels.csv")
TRAIN_ROOT = os.path.join(BASE_PATH, "DATASET", "train")
VAL_ROOT = os.path.join(BASE_PATH, "DATASET", "test")
LOG_FILE_PATH = os.path.join(SAVE_DIR, "training_log_5token.txt")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "best_frit_weights_5token_15.pth")
PLOT_PATH = os.path.join(SAVE_DIR, "training_results_plot_5token.png")