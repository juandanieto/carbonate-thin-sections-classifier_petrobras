from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.models as models
from tqdm.auto import tqdm
import time

from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path

# --- GLOBAL CONSTANTS ---
TRAIN_DIR = "/home/hackathon/Lamina/train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT_RATIO = 0.2
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4

# Normalization constants (Standard ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 1. --- Exploratory Data Analysis ---

def analyze_black_dot_artifacts(source_dir ="/home/hackathon/Lamina/train", min_pixel_threshold=5):
    """
    Analyzes images in a directory for the presence of dark artifacts (black dots/patches)
    by checking the minimum pixel value. Generates a report and a stacked bar chart 
    showing the breakdown per class.
    """
    
    lithology_breakdown = defaultdict(lambda: {'has_black_balls': 0, 'no_black_balls': 0})
    total_scanned = 0

    if not os.path.isdir(source_dir):
        print(f"Error: Directory not found at {source_dir}")
        return

    for filename in os.listdir(source_dir):
        if not filename.endswith('.png'):
            continue

        total_scanned += 1
        source_path = os.path.join(source_dir, filename)

        label = "Unknown"
        try:
            parts = filename.split('_')
            if len(parts) > 4:
                label = parts[4]
        except Exception:
            pass 

        has_black_balls = False
        try:
            with Image.open(source_path) as img:
                img_gray = img.convert('L')
                img_array = np.array(img_gray, dtype=np.uint8)
                
                if np.min(img_array) < min_pixel_threshold:
                    has_black_balls = True

        except Exception:
            continue

        if has_black_balls:
            lithology_breakdown[label]['has_black_balls'] += 1
        else:
            lithology_breakdown[label]['no_black_balls'] += 1

    # 2. Plotting (Stacked Bar Chart by Class) 

    if not lithology_breakdown:
        print("No data found to plot.")
        return

    # Prepare DataFrame for plotting
    df = pd.DataFrame(lithology_breakdown).fillna(0).astype(int).transpose()
    df.columns = ['With Black Dots', 'Without Black Dots']
    
    if 'Unknown' in df.index and df.loc['Unknown'].sum() == 0:
        df = df.drop('Unknown')

    df = df[['With Black Dots', 'Without Black Dots']]
    df['Total'] = df.sum(axis=1)

    # Create the stacked bar chart
    ax = df[['With Black Dots', 'Without Black Dots']].plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 7), 
        color={'With Black Dots': 'black', 'Without Black Dots': 'lightgray'}
    )

    plt.title('Black Dot Artifact Distribution by Lithology Class', fontsize=16)
    plt.xlabel('Lithology Class Label', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Artifact Presence')
    
    # Logic for Adding Total Labels on top of each bar
    for i, container in enumerate(ax.containers):
        # Add labels for the individual stacks
        is_bottom_stack = (i == 0)
        
        for patch in container.patches:
            height = patch.get_height()
            
            if height > 0: # Only label non-zero stacks
                # Calculate the center position of the patch
                x_center = patch.get_x() + patch.get_width() / 2.
                y_pos = patch.get_y() + height / 2.
                
                # Add the count label (individual stack count)
                ax.text(x_center, y_pos, str(int(height)), ha='center', va='center', fontsize=9, color='white' if is_bottom_stack else 'black')

    # Logic for Adding Grand Total Labels on top
    for i, patch in enumerate(ax.containers[-1].patches):
        total_value = df['Total'].iloc[i]
        
        # Add the total label slightly above the top of the bar
        ax.text(patch.get_x() + patch.get_width() / 2., 
                patch.get_y() + patch.get_height(), 
                str(int(total_value)), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.show()


def plot_class_samples(data_dir="/home/hackathon/Lamina/train", n_samples=5):
    """
    Scans a directory, discovers labels from filenames (e.g., ..._LABEL_...),
    and plots a grid of 'n_samples' images for each discovered class.
    """
    files_by_label = defaultdict(list)
    
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return

    # 1. Discover and group all files by their label
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            try:
                # Get label from 5th part (index 4)
                parts = filename.split('_')
                if len(parts) > 4:
                    label = parts[4]
                    files_by_label[label].append(filename)
            except Exception:
                pass

    if not files_by_label:
        print("No files with valid labels found in the directory.")
        return
        
    # 2. Plot the grid
    n_classes = len(files_by_label)
    
    plt.figure(figsize=(n_samples * 3, n_classes * 3))
    
    sorted_labels = sorted(files_by_label.keys())
    
    for i, label in enumerate(sorted_labels):
        filenames = files_by_label[label]
        
        for j in range(n_samples):
            
            if j < len(filenames):
                img_path = os.path.join(data_dir, filenames[j])
                
                try:
                    img = Image.open(img_path)
                    
                    ax = plt.subplot(n_classes, n_samples, i * n_samples + j + 1)
                    plt.imshow(img)
                    plt.axis("off")
                    
                    if j == 0:
                        ax.set_title(label, fontsize=14, loc='left', fontweight='bold')
                        
                except Exception:
                    ax = plt.subplot(n_classes, n_samples, i * n_samples + j + 1)
                    plt.text(0.5, 0.5, "Error", ha='center', va='center')
                    plt.axis("off")
            else:
                ax = plt.subplot(n_classes, n_samples, i * n_samples + j + 1)
                plt.axis("off")

    plt.suptitle("Sample Images per Lithology Class", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

#2. -- Preprocessing --

def get_sampler(labels):
    """Calculates class weights and returns a WeightedRandomSampler."""
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    # Map class index to frequency
    class_weights_dict = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Get weights for each sample
    sample_weights = [class_weights_dict[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    return sampler, class_weights_dict

#Dataset Class
class LithologyDataset(Dataset):
    """Custom PyTorch Dataset using pre-split lists of file paths and labels."""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image) 

            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception:
            # Handle corrupted files by loading a random non-corrupted sample
            return self.__getitem__(random.randint(0, len(self.file_paths) - 1)) 

def create_dataloaders(val_split_ratio, img_size, batch_size, IMAGENET_MEAN = [0.485, 0.456, 0.406],IMAGENET_STD = [0.229, 0.224, 0.225],
seed=42,data_dir="/home/hackathon/Lamina/train"):
    """
    Scans the directory for image files, splits data, calculates and adjusts 
    class weights, and returns the DataLoaders and associated metadata.
    """
    # Initialize lists for paths and text labels
    full_paths = []
    labels_text = []

    # 1. Initial File Discovery and Label Extraction
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            try:
                labels_text.append(filename.split('_')[4])
                full_paths.append(os.path.join(data_dir, filename))
            except IndexError:
                # Skip files that don't match the expected naming convention
                continue

    # 2. Dynamic Label Mapping
    unique_labels = sorted(list(set(labels_text)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_map)
    labels_encoded = [label_map[text_label] for text_label in labels_text]

    # 3. Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        full_paths,
        labels_encoded,
        test_size=val_split_ratio,
        random_state=seed,
        stratify=labels_encoded
    )

    # 4. Define Image Transformations
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomApply([
            T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation([10, -1]),
        ], p=0.7),
        T.RandomApply([
            T.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.5),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 5. Create Datasets and Sampler/Weights
    train_dataset = LithologyDataset(X_train, y_train, transform=train_transform)
    val_dataset = LithologyDataset(X_val, y_val, transform=val_transform)

    # Get initial class weights and sampler (from helper function)
    sampler, class_weights_dict = get_sampler(y_train)
    class_weights = torch.tensor(
        [class_weights_dict[i] for i in range(num_classes)],
        dtype=torch.float32
    )

    # Apply Penalization/Boosting Factors
    ESF_INDEX = label_map.get('ESF')
    LMT_INDEX = label_map.get('LMT')
    ETR_INDEX = label_map.get('ETR')

    if ESF_INDEX is not None:
        class_weights[ESF_INDEX] = class_weights[ESF_INDEX] * 2.0

    if LMT_INDEX is not None:
        class_weights[LMT_INDEX] = class_weights[LMT_INDEX] * 2.0

    if ETR_INDEX is not None:
        class_weights[ETR_INDEX] = class_weights[ETR_INDEX] * 1.2

    print(f"Adjusted Class Weights: {class_weights}")

    # 6. Create DataLoaders
    # Train Loader uses the sampler for balanced batch selection (shuffle=False when sampler is used)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Validation Loader uses standard sequential access
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, class_weights, num_classes, label_map

    
# 3. -- Modeling ---

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one full training pass over the dataset."""
    model.train()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    batch_pbar = tqdm(dataloader, desc='Training Batch', leave=False)
    for inputs, labels in batch_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    """Performs one full validation pass over the dataset."""
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def plot_training_history(history):
    """Plots the training and validation loss and accuracy."""

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_combined_history(history, ft_start_epoch):
    """Plots combined history with a line indicating the start of fine-tuning."""
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=ft_start_epoch -1, color='red', linestyle='--', label='Fine-Tuning Start')
    plt.title('Loss (Feature Extraction + Fine-Tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.axvline(x=ft_start_epoch -1, color='red', linestyle='--', label='Fine-Tuning Start')
    plt.title('Accuracy (Feature Extraction + Fine-Tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()

def set_seed(seed):
    # Set seed for Python's built-in random number generator
    random.seed(seed)

    # Set seed for NumPy's random number generator (if used)
    np.random.seed(seed)

    # Set seed for PyTorch's CPU random number generator
    torch.manual_seed(seed)

    # Set seed for PyTorch's CUDA random number generator (if using GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    # For ensuring deterministic behavior in some CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Recommended to set to False with deterministic=True


#4. -- Results --

def evaluate_model_performance(model, dataloader, device, description):
    """Runs a forward pass and collects true labels and predictions."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=description):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def display_metrics(true_labels, predicted_labels, class_names, title_suffix):
    """Calculates and displays F1 score and Confusion Matrix for a given set."""

    # Calculate F1 scores per class
    f1_scores_per_class = f1_score(
        true_labels, predicted_labels, labels=range(len(class_names)),
        average=None, zero_division=0
    )

    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(
        true_labels, predicted_labels, labels=range(len(class_names))
    )

    print(f"\n--- Final {title_suffix} Metrics ---")
    print("F1-Score per Class:")

    for label, f1 in zip(class_names, f1_scores_per_class):
        print(f"  {label}: {f1:.4f}")

    print(f"Macro Average F1 Score: {f1_score(true_labels, predicted_labels, average='macro'):.4f}")

    # Plotting Confusion Matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix_df, annot=True, fmt='d', cmap='Blues',
        cbar=True, linewidths=.5, linecolor='black'
    )
    plt.title(f'{title_suffix} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def run_evaluation_pipeline(model, train_loader, val_loader, device, model_path, class_labels):
    """
    Loads the final fine-tuned model weights, sets the model to evaluation mode, 
    and executes performance evaluation (predictions, metrics) on both 
    Training and Validation sets.
    """

    print("--- Final Evaluation Setup ---")

    # Load the best fine-tuned weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded BEST model weights from {model_path}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {model_path}. Check training directory.")
        return

    # Set the model to evaluation mode (crucial for Dropout and BatchNorm)
    model.eval()

    # Get true labels and predictions for both sets (using helper function)
    with torch.no_grad():
        train_true, train_preds = evaluate_model_performance(model, train_loader, device, "Evaluating Training Set")
        val_true, val_preds = evaluate_model_performance(model, val_loader, device, "Evaluating Validation Set")

    # Display Classification Metrics (using helper function)
    print("\n" + "="*50)
    display_metrics(train_true, train_preds, class_labels, "TRAINING SET")
    print("\n" + "="*50)
    display_metrics(val_true, val_preds, class_labels, "VALIDATION SET")
    print("="*50 + "\n")

#5. -- Final Prediction and Submission Export (Submitted) --

def extract_collate_fn_ignore_none(batch):
    """Custom collate function to filter out failed samples (None) and keep indices."""
    # Filter out samples where the image is None (failed to load)
    # Batch is now a list of (image_tensor, index)
    batch = [b for b in batch if b[0] is not None] 
    
    if not batch:
        return None, None # Return None if the entire batch failed
        
    return torch.utils.data.dataloader.default_collate(batch)


def create_test_manifest(source_dir, output_manifest_path):
    """Scans the test directory and creates a manifest with filepath and filename."""
    # ... (Function remains unchanged)
    # [Rest of create_test_manifest function is assumed correct]
    print(f"Scanning test directory {source_dir}...")
    file_data = []
    
    if not os.path.exists(source_dir):
        print(f"Error: Test directory not found: {source_dir}")
        return False

    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            filepath = os.path.join(source_dir, filename)
            file_data.append((filepath, filename))

    df = pd.DataFrame(file_data, columns=['filepath', 'filename'])
    df.to_csv(output_manifest_path, index=False)
    print(f"Test manifest created: {output_manifest_path} (Entries: {len(df)})")
    return True

class TestDataset(Dataset):
    """Custom Dataset for loading test images for inference."""
    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.filename_list = self.df['filename'].tolist()
        self.filename_origin = self.df['image_filename'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return None, idx for failed samples. collate_fn will filter it.
            return None, idx 

        if self.transform:
            image = self.transform(image)
            
        # Returns image tensor and its original index in the dataframe
        return image, idx 


def make_predictions(model, dataloader, device, patches=True):
    """
    Runs inference on the test set and returns predictions and filenames.
    If patches=True, aggregates patch-level predictions by image.
    """
    model.eval()
    all_preds = []
    processed_indices = []
    
    all_filenames = dataloader.dataset.filename_list
    all_origins = dataloader.dataset.filename_origin # original image (without patch)
    
    print("Starting final inference...")
    with torch.no_grad():
        for inputs, indices in tqdm(dataloader, desc="Predicting Test Set"):
            if inputs is None:
                continue
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            processed_indices.extend(indices.cpu().numpy())
    
    # Reconstruct lists based on processed indices
    test_ids = [all_filenames[i] for i in processed_indices]
    image_origins = [all_origins[i] for i in processed_indices]
    
    # Map int → label string
    inv_label_map = {i: label for label, i in LABEL_MAP.items()} # LABEL_MAP must be defined globally
    final_labels = [inv_label_map[p] for p in all_preds]
    
    # --- Aggregate by image if patches=True ---
    if patches:
        df_preds = pd.DataFrame({
            "patch_filename": test_ids,
            "original_image": image_origins,
            "pred_label": final_labels
        })

        # group by image and take the most frequent class
        agg_preds = (
            df_preds.groupby("original_image")["pred_label"]
            .agg(lambda x: Counter(x).most_common(1)[0][0])
            .reset_index()
        )

        # replace outputs with the aggregated results
        test_ids = agg_preds["original_image"].tolist()
        final_labels = agg_preds["pred_label"].tolist()

        print(f"\n Aggregated {len(df_preds)} patch predictions → {len(agg_preds)} image predictions.")
    
    return test_ids, final_labels

def execute_submission_pipeline():
    """Runs the full test inference and exports the submission file."""

    print("--- Final Prediction and Submission Export ---")

    # 1. Create Test Manifest
    if not os.path.exists(TEST_MANIFEST_PATH):
        if not create_test_manifest(TEST_DIR, TEST_MANIFEST_PATH):
            print("Error: Cannot proceed without test files.")
            return
    else:
        print(f"Test manifest already exists at {TEST_MANIFEST_PATH}. Skipping creation.")

    # 2. Setup Test DataLoader
    test_transform = T.Compose([
        #T.Resize((IMG_SIZE, IMG_SIZE)), # IMG_SIZE must be defined globally
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # IMAGENET_MEAN/STD must be defined globally
    ])

    test_dataset = TestDataset(manifest_csv=TEST_MANIFEST_PATH, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=extract_collate_fn_ignore_none,
        pin_memory=True
    )

    # 3. Load the Best Model Weights
    
    # CRITICAL CORRECTION: Re-initialize the ResNeXt-50 architecture (32x4d)
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    # FIX: Recreate the sequential layer with the trained dropout layer (assumed to be 1D with 0.15 rate)
    # This prevents the "Missing key(s)" error during state_dict loading.
    model.fc = nn.Sequential(
        nn.Dropout1d(p=0.2), # Use nn.Dropout1d as seen in the training code provided
        nn.Linear(num_ftrs, NUM_CLASSES) # NUM_CLASSES must be defined globally
    )
    
    # Load weights
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE)) # DEVICE must be defined globally
    model.to(DEVICE)
    print(f"Successfully loaded model weights from {BEST_MODEL_PATH}")

    # 4. Make Predictions
    test_ids, predictions = make_predictions(model, test_loader, DEVICE)

    # 5. Create Submission DataFrame with REQUIRED column names
    submission_df = pd.DataFrame({
        'img': test_ids,
        'classe': predictions 
    })

    # 6. Save and Print Results
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)

    print(f"\nSubmission file created successfully: {SUBMISSION_FILE_PATH} (Entries: {len(submission_df)})")

    print("\n--- First 5 lines of Submission ---")
    print(submission_df.head())


#8. -- EXTRA: AI Inpainting --

def check_for_black_dots(image_path: Path, threshold: int = 5, min_pixel_count: int = 1000):
    """
    Detects if an image contains a large black hole artifact.
    """
    try:
        # Read the image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            return False
            
        # Count pixels close to black (value <= threshold)
        black_pixel_count = np.sum(img <= threshold)
            
        # Only return True if the count exceeds the large threshold
        return black_pixel_count > min_pixel_count
    except Exception as e:
        print(f"Error checking {image_path}: {e}")
        return False

def create_mask_from_image(image_pil: Image.Image):
    """Creates the binary mask (white hole) from the black artifacts."""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Identify very dark pixels (potential artifacts)
    mask_cv = cv2.inRange(gray, 0, 10) 
        
    # Dilate mask slightly to cover edges better
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask_cv, kernel, iterations=1)
        
    return Image.fromarray(mask_dilated)


def run_inpainting_pipeline(manifest_path: Path, original_dir: Path, new_dir: Path, model_id: str, device: str):
    """Orchestrates model loading, artifact detection, inpainting, and file saving."""

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found. Cannot proceed.")
        return

    # 1. Setup Output Directory
    new_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created at: {new_dir}")

    # 2. Load AI Model
    print(f"Loading Inpainting model ({model_id}) onto the GPU...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    print("Model loaded.")

    # 3. Processing Loop
    df = pd.read_csv(manifest_path)
    print(f"Processing {len(df)} images. This will take time...")
    
    images_corrected = 0
    images_copied = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        # Assuming 'path' in manifest is relative to ORIGINAL_SOURCE_DIR, or is full path.
        # Using row['path'] directly if it's the full path, otherwise uncomment next line:
        image_path = Path(row['path']) 
        image_name = image_path.name
        output_path = new_dir / image_name

        # 1. Check if the image has a large black hole
        if check_for_black_dots(image_path):
            
            # 2. Execute AI inpainting
            images_corrected += 1
            image = Image.open(image_path).convert("RGB")
                
            # Resize image and mask to SD input size (512x512)
            image_512 = image.resize((IMG_SIZE_INPAINT, IMG_SIZE_INPAINT))
            mask_512 = create_mask_from_image(image_512)
                
            # Run the Stable Diffusion Inpainting pipeline
            inpainted_image = pipe(
                prompt=PROMPT, 
                image=image_512, 
                mask_image=mask_512,
                # num_inference_steps=20, # Uncomment for faster (lower quality) results
            ).images[0]
                
            # Resize back to training size and save
            inpainted_image.resize((IMG_SIZE_FINAL, IMG_SIZE_FINAL)).save(output_path)
        else:
            # 3. If no large hole is detected, copy the original file
            images_copied += 1
            shutil.copy(image_path, output_path)

    print("\n--- Offline Processing Complete! ---")
    print(f"AI Corrected Images (with holes): {images_corrected}")
    print(f"Images Copied (clean or noisy): {images_copied}")
    print(f"All images are ready in {new_dir}")


# 1. Black Dot Detection Function 
def check_for_black_dots(image_path, threshold=5):
    """Detects if an image contains near-black pixels."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        black_pixel_count = np.sum(img <= threshold)
        # Using a low threshold (10 pixels) to reliably find dot samples for verification
        return black_pixel_count > 10 
    except Exception:
        return False


# 2. Verification Logic 
def visualize_inpainting_correction(original_manifest_path, corrected_source_dir, num_samples=10):
    """Loads dot-containing images and plots them against their AI-corrected versions."""
    
    print(f"Loading original manifest: {original_manifest_path}")

    if not os.path.exists(original_manifest_path):
        print(f"Error: Original manifest '{original_manifest_path}' not found.")
        return

    if not os.path.exists(corrected_source_dir):
        print(f"Error: Corrected directory not found at {corrected_source_dir}.")
        return

    df = pd.read_csv(original_manifest_path)
    
    # 1. Find samples that originally had black dots
    print("Searching for 10 original images with black dots...")
    df['has_dots'] = df['path'].apply(check_for_black_dots)
    dot_df = df[df['has_dots'] == True]
    
    if len(dot_df) >= num_samples:
        samples_df = dot_df.sample(n=num_samples, random_state=42)
    else:
        print(f"Alert: Only found {len(dot_df)} dot-images. Showing all found.")
        samples_df = dot_df

    if samples_df.empty:
        print("No images with black dots were found for verification.")
        return
    
    # Simple resize transform for plotting consistency
    transform = T.Compose([
        T.Resize((256, 256)), 
    ])

    # 2. Plotting
    num_samples = len(samples_df)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    fig.suptitle("AI Inpainting Verification [Original (Left) | Corrected (Right)]", fontsize=16, y=1.02)
    
    if num_samples == 1:
        axes = np.array([axes])

    for i, (index, row) in enumerate(samples_df.iterrows()):
        original_path = row['path']
        file_name = os.path.basename(original_path)
        corrected_path = os.path.join(corrected_source_dir, file_name)
        
        try:
            original_img = Image.open(original_path).convert("RGB")
            corrected_img = Image.open(corrected_path).convert("RGB")
            
            original_transformed = transform(original_img)
            corrected_transformed = transform(corrected_img)
            
            # Column 1: Original
            axes[i, 0].imshow(original_transformed)
            axes[i, 0].set_title(f"{file_name} (Original)", fontsize=8)
            axes[i, 0].axis('off')
            
            # Column 2: Corrected by AI
            axes[i, 1].imshow(corrected_transformed)
            axes[i, 1].set_title(f"{file_name} (AI Corrected)", fontsize=8)
            axes[i, 1].axis('off')
            
        except FileNotFoundError:
            print(f"Error: Corrected image not found at {corrected_path}. Skipping sample.")
            if num_samples > 1:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def initialize_final_model(dropout_rate: float, num_classes: int, device: torch.device):
    """Re-initializes the ResNeXt-50 architecture to match the trained structure (Dropout + Linear)."""
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # CRITICAL: Recreate the sequential layer to ensure compatibility with saved weights
    model.fc = nn.Sequential(
        nn.Dropout1d(p=dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    model.to(device)
    return model











