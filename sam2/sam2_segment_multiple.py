import os
# Set environment variables for CUDA settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0
os.environ['CUDA_MANAGED_FORCE_DEVICE_ALLOC'] = '1'  # Enable unified memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import cv2
import numpy as np
import torch
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from shapely.geometry import Polygon, MultiPolygon
from supervision.draw.color import Color, ColorPalette
import supervision as sv
from pdf2image import convert_from_path  # For PDF processing
from torch.cuda.amp import autocast

# --- Configuration ---

# Replace with absolute paths
folder1_path = 'aalesund_fokus'         # Folder containing 25 images
folder2_path = 'aalesund_utfordring'    # Folder containing 10 images

# Output folders
output_folder = 'aalesund_segmented_images'
output_folder_fokus = os.path.join(output_folder, 'fokus')
output_folder_utfordring = os.path.join(output_folder, 'utfordring')

# Create output directories if they don't exist
os.makedirs(output_folder_fokus, exist_ok=True)
os.makedirs(output_folder_utfordring, exist_ok=True)

# Paths to configuration and checkpoint files
#config_file_path = './sam2.1_hiera_l.yaml'      
#checkpoint_file_path = './sam2.1_hiera_large.pt' 

config_file_path = './sam2.1_hiera_b+.yaml'     
checkpoint_file_path = './sam2.1_hiera_base_plus.pt' 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Initialize the SAM2 Model ---

GlobalHydra.instance().clear()

with initialize(config_path="."):
    print("Initializing the SAM2 model...")
    sam2_model = build_sam2(config_file=config_file_path, ckpt_path=checkpoint_file_path).to(device)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=64,
    points_per_batch=16,
    pred_iou_thresh=0.0,
    stability_score_thresh=0.85,
    stability_score_offset=1.0,
    mask_threshold=0.0,
    box_nms_thresh=1,           
    crop_n_layers=1,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=0.2,        
    crop_n_points_downscale_factor=1,
    point_grids=None,
    min_mask_region_area=0,
    output_mode="binary_mask",
    use_m2m=True,                  
    multimask_output=False
)

print("Model and mask generator initialized successfully.")

# --- Helper Functions ---

def is_mask_inside(outer_mask, inner_mask):
    # Check if all True pixels in inner_mask are also True in outer_mask
    return np.all(outer_mask[inner_mask > 0])

def custom_mode(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

def get_most_common_color(image_bgr, mask):
    mask_area = np.where(mask)
    pixels = image_bgr[mask_area]
    if pixels.size == 0 or pixels.ndim != 2 or pixels.shape[1] != 3:
        return (0, 0, 0)
    # Use custom mode for each channel (BGR order)
    b_mode = int(custom_mode(pixels[:, 0]))
    g_mode = int(custom_mode(pixels[:, 1]))
    r_mode = int(custom_mode(pixels[:, 2]))
    return (b_mode, g_mode, r_mode)

def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
    for existing_poly in existing_polys:
        intersection_area = poly.intersection(existing_poly).area
        if poly.area == 0:
            continue
        overlap_ratio = intersection_area / poly.area
        if overlap_ratio >= area_overlap_threshold:
            return True
    return False

def pdf_to_images(pdf_path):
    # Convert PDF pages to images
    try:
        images = convert_from_path(pdf_path)  # Add poppler_path if necessary
        image_bgr_list = []
        for img in images:
            image_rgb = np.array(img)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_bgr_list.append(image_bgr)
        return image_bgr_list
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

# --- Processing Function ---

def process_image(image_path, output_path_base):
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        return

    if image_path.lower().endswith('.pdf'):
        image_bgr_list = pdf_to_images(image_path)
        if not image_bgr_list:
            print(f"No images extracted from PDF: {image_path}")
            return
    else:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Failed to load image: {image_path}")
            return
        else:
            print(f"Image loaded successfully: {image_path}")
        image_bgr_list = [image_bgr]

    page_num = 1
    for image_bgr in image_bgr_list:

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate masks
        with autocast():
            sam_result = mask_generator.generate(image_rgb)

        # Filter masks to remove smaller masks inside larger ones
        # Step 1: Extract masks and compute area for sorting
        masks_with_areas = []
        for i, mask in enumerate(sam_result):
            segmentation = mask['segmentation'].astype(bool)
            if np.any(segmentation):
                area = np.sum(segmentation)
                masks_with_areas.append((i, segmentation, area))

        if not masks_with_areas:
            print(f"No valid masks found for image: {image_path}")
            continue

        # Sort masks by area (from largest to smallest)
        masks_with_areas.sort(key=lambda x: x[2], reverse=True)

        # Identify masks to remove
        indices_to_remove = set()

        num_masks = len(masks_with_areas)
        # Loop through masks and remove larger masks that contain multiple smaller masks
        for i, (outer_idx, outer_mask, outer_area) in enumerate(masks_with_areas):
            contained_count = 0
            for inner_idx, inner_mask, inner_area in masks_with_areas[i+1:]:
                if outer_mask.shape != inner_mask.shape:
                    print(f"Warning: Masks have different shapes: outer_mask {outer_mask.shape}, inner_mask {inner_mask.shape}")
                    continue
                try:
                    if is_mask_inside(outer_mask, inner_mask):
                        contained_count += 1
                except Exception as e:
                    print(f"Error in is_mask_inside: {e}")
                    continue
            if contained_count >= int(0.5 * num_masks):
                indices_to_remove.add(outer_idx)

        # Filter out the unwanted masks
        filtered_masks_with_areas = [
            (idx, mask, area) for idx, mask, area in masks_with_areas if idx not in indices_to_remove
        ]

        # Also remove any masks that cover the entire image
        image_area = image_bgr.shape[0] * image_bgr.shape[1]
        filtered_masks_with_areas = [
            (idx, mask, area) for idx, mask, area in filtered_masks_with_areas if area < image_area
        ]

        if not filtered_masks_with_areas:
            print(f"No masks left after filtering for image: {image_path}")
            continue

        # Create a filtered sam_result
        filtered_sam_result = [sam_result[idx] for idx, _, _ in filtered_masks_with_areas]

        polygons_list = []

        # Prepare a copy of the original image for drawing polygons
        image_with_polygons = image_bgr.copy()
        
        # Image area
        image_area = image_bgr.shape[0] * image_bgr.shape[1]
        
        # List to store polygons with their area
        mask_polygons = []
        
        # Function to smooth contour using moving average
        def smooth_contour(contour, window_size=5):
            # Ensure window_size is odd
            if window_size % 2 == 0:
                window_size += 1
            half_window = window_size // 2
        
            # Pad the contour to handle the circular nature
            contour = np.concatenate((contour[-half_window:], contour, contour[:half_window]), axis=0)
            
            smoothed_contour = []
            for i in range(half_window, len(contour) - half_window):
                window_points = contour[i - half_window:i + half_window + 1]
                mean_point = np.mean(window_points, axis=0)
                smoothed_contour.append(mean_point)
            smoothed_contour = np.array(smoothed_contour, dtype=np.int32)
            return smoothed_contour
        
        # Loop over each mask in the filtered SAM result
        for idx, mask_dict in enumerate(filtered_sam_result):
            mask = mask_dict['segmentation'].astype(np.uint8)  # Ensure mask is in uint8 format
        
            # Find contours in the mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
            # Skip if no contours are found
            if not contours:
                continue
        
            # Process each contour
            for contour in contours:
                if contour.shape[0] < 5:
                    continue  # Need at least 5 points to smooth
        
                # Reshape contour to 2D array
                contour = contour.reshape(-1, 2)
        
                # Smooth the contour using moving average
                smoothed_contour = smooth_contour(contour, window_size=15)  # Adjust window_size as needed
        
                if smoothed_contour.shape[0] >= 3:
                    polygon = Polygon(smoothed_contour)
                    # Ensure the polygon is valid
                    if not polygon.is_valid or polygon.area == 0:
                        # Try fixing invalid polygons
                        polygon = polygon.buffer(0)
                        if not polygon.is_valid or polygon.area == 0:
                            continue  # Skip if still invalid
                    # Store the polygon along with its area and index
                    mask_polygons.append({'area': polygon.area, 'polygon': polygon, 'index': idx})
        
        # Introduce max_area_threshold to exclude overly large polygons
        max_area_threshold = 0.9 * image_area  # Exclude polygons covering more than 90% of the image
        
        # Filter out masks that are too large
        mask_polygons = [mp for mp in mask_polygons if mp['area'] < max_area_threshold]
        
        # Debug: Print the number of polygons after excluding large masks
        print(f"Total polygons after excluding large masks: {len(mask_polygons)}")
        
        # Now, filter out smaller polygons that are mostly within larger ones
        # Sort polygons by area in descending order
        mask_polygons.sort(key=lambda x: x['area'], reverse=True)
        
        # Initialize list to hold the final polygons
        filtered_polygons = []
        
        # Function to check if a polygon is mostly within existing polygons
        def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
            for existing_poly in existing_polys:
                intersection_area = poly.intersection(existing_poly).area
                if poly.area == 0:
                    continue
                overlap_ratio = intersection_area / poly.area
                if overlap_ratio >= area_overlap_threshold:
                    return True
            return False
        
        # Process each polygon
        for idx, poly_dict in enumerate(mask_polygons):
            poly = poly_dict['polygon']
            if not is_polygon_mostly_within(poly, [d['polygon'] for d in filtered_polygons], area_overlap_threshold=0.2):
                filtered_polygons.append(poly_dict)
            else:
                print(f"Polygon {idx} is mostly within another polygon and will be removed.")
        
        # Debug: Print the number of polygons after filtering
        print(f"Total polygons after overlap filtering: {len(filtered_polygons)}")
        
        # Draw the filtered polygons on the image
        for poly_dict in filtered_polygons:
            poly = poly_dict['polygon']
            if isinstance(poly, Polygon):
                # Handle single Polygon
                coords = np.array(list(poly.exterior.coords)).astype(np.int32)
                cv2.polylines(image_with_polygons, [coords], isClosed=True, color=(0, 255, 0), thickness=5)
                polygons_list.append(poly)
            elif isinstance(poly, MultiPolygon):
                # Handle MultiPolygon
                for sub_poly in poly.geoms:
                    if sub_poly.is_valid and not sub_poly.is_empty:
                        coords = np.array(list(sub_poly.exterior.coords)).astype(np.int32)
                        cv2.polylines(image_with_polygons, [coords], isClosed=True, color=(0, 255, 0), thickness=5)
                        polygons_list.append(sub_poly)
        
            # Save the image with polygons
            if len(image_bgr_list) > 1:
                output_path = f"{output_path_base}_page{page_num}.png"
            else:
                output_path = f"{output_path_base}.png"
            cv2.imwrite(output_path, image_with_polygons)
            print(f"Saved polygonized image to {output_path}")
            page_num += 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def collect_image_paths(folder_path, max_images):
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.pdf')
    all_files = os.listdir(folder_path)
    image_files = []
    for f in all_files:
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            if f.lower().endswith(image_extensions):
                image_files.append(file_path)
            else:
                print(f"Skipping unsupported file extension: {file_path}")
        else:
            print(f"Skipping non-file: {file_path}")
    return image_files[:max_images]

# Process images from the first folder (25 images)
folder1_images = collect_image_paths(folder1_path, max_images=25)
for image_path in folder1_images:
    filename = os.path.basename(image_path)
    if filename == "240.jpg":
        continue
    filename_base, _ = os.path.splitext(filename)
    # Update output path to save into 'fokus' subfolder
    output_path_base = os.path.join(output_folder_fokus, f"polygonized_{filename_base}")
    process_image(image_path, output_path_base)

# Process images from the second folder (10 images)
folder2_images = collect_image_paths(folder2_path, max_images=10)
for image_path in folder2_images:
    filename = os.path.basename(image_path)
    filename_base, _ = os.path.splitext(filename)
    # Update output path to save into 'utfordring' subfolder
    output_path_base = os.path.join(output_folder_utfordring, f"polygonized_{filename_base}")
    process_image(image_path, output_path_base)
