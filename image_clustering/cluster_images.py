import os
import cv2
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torchvision import models, transforms
import torch
from shutil import copy2
from pdf2image import convert_from_path

# Paths to subdirectories within the main directory
input_base = 'aalesund'
input_fokus = os.path.join(input_base, 'FOKUS')
input_utfordring = os.path.join(input_base, 'UTFORDRING')

# Paths to output clustered directories
cluster_output_base = 'aalesund_clustered_images'
os.makedirs(cluster_output_base, exist_ok=True)

# Feature extraction using ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True).to(device)  # Use a deeper ResNet model
resnet.eval()
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_deep_features(image):
    """Extract deep features using ResNet."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features.cpu().numpy().flatten()

def pdf_to_images(pdf_path):
    """Converts PDF pages to images."""
    try:
        images = convert_from_path(pdf_path)
        image_bgr_list = []
        for img in images:
            image_rgb = np.array(img)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_bgr_list.append(image_bgr)
        return image_bgr_list
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def collect_images_and_features_recursive(base_path):
    """Collects images and their features from subdirectories within base_path, filtering files based on specific rules."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pdf')
    images = []
    features = []

    # Define patterns
    exclude_pattern = re.compile(r'_planbestemmelser', re.IGNORECASE)  # Exclude specific files
    include_pattern = re.compile(r'^[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)?$', re.IGNORECASE)  # Allow "letters and numbers only"

    # Walk through all subdirectories starting from the given base_path
    for root, _, files in os.walk(base_path):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_extensions) and not exclude_pattern.search(file_lower):
                if include_pattern.match(file) or "plankart" in file_lower:
                    file_path = os.path.join(root, file)
                    if file_path.lower().endswith('.pdf'):
                        image_list = pdf_to_images(file_path)
                        for image in image_list:
                            feature = extract_deep_features(image)
                            if feature is not None:
                                images.append(image)
                                features.append(feature)
                    else:
                        image = cv2.imread(file_path)
                        if image is None:
                            print(f"Failed to load image: {file_path}")
                            continue
                        feature = extract_deep_features(image)
                        if feature is not None:
                            images.append(file_path)
                            features.append(feature)
    return images, features

def reduce_dimensionality(features, n_components=50):
    """Reduce feature dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    print(f"Reduced features to {n_components} dimensions, explaining {sum(pca.explained_variance_ratio_):.2%} variance.")
    return reduced_features

def find_best_k(features, max_k=10):
    """Determines the optimal number of clusters using Silhouette Score."""
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))
    
    best_k = K[np.argmax(silhouette_scores)]
    print(f"Best k found: {best_k} with silhouette score: {max(silhouette_scores):.4f}")
    return best_k

def cluster_and_save(images, features, n_clusters):
    """Clusters images and saves them into separate directories."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(features)
    
    for cluster_idx in range(n_clusters):
        cluster_folder = os.path.join(cluster_output_base, f'cluster_{cluster_idx}')
        os.makedirs(cluster_folder, exist_ok=True)
    
    for img_path, label in zip(images, labels):
        cluster_folder = os.path.join(cluster_output_base, f'cluster_{label}')
        if isinstance(img_path, str):  # If it's a file path
            copy2(img_path, cluster_folder)
        else:  # If it's an image array
            output_path = os.path.join(cluster_folder, f"image_{label}.jpg")
            cv2.imwrite(output_path, img_path)

# Collect and extract features from subdirectories within fokus and utfordring
all_images = []
all_features = []

for base_path in [input_fokus, input_utfordring]:
    images, features = collect_images_and_features_recursive(base_path)
    all_images.extend(images)
    all_features.extend(features)

# Check if features are extracted
if not all_features:
    print("No features extracted. Check if the input folders contain valid images or PDFs.")
    exit()

# Convert features to a NumPy array
all_features = np.array(all_features)

# Reduce dimensionality
all_features_reduced = reduce_dimensionality(all_features, n_components=50)

# Find the best k
best_k = find_best_k(all_features_reduced, max_k=15)

# Perform clustering with the best k
cluster_and_save(all_images, all_features_reduced, best_k)

print(f"Images (including PDFs) have been clustered into {best_k} groups under '{cluster_output_base}'.")
