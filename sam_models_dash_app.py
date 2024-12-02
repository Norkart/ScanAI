import dash
from dash import dcc, html, Input, Output, State
import cv2
import numpy as np
import plotly.express as px
from samgeo import SamGeo2, SamGeo
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from supervision.draw.color import Color
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import base64
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from samgeo.hq_sam import SamGeo as SamGeoHQ


# Load the image
image_bgr = cv2.imread('datasets/aalesund/1504200/200.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Initialize the app
app = dash.Dash(__name__)
app.title = "Automatic Mask Generator"

# App layout
app.layout = html.Div([
    html.H1("Automatic Mask Generator", style={'textAlign': 'center'}),
    dcc.RadioItems(
        id='mask-generator',
        options=[
            {'label': 'SAM', 'value': 'sam'},
            {'label': 'SAM2', 'value': 'sam2'},
            {'label': 'GEOSAM', 'value': 'geosam'},
            {'label': 'GEOSAM2', 'value': 'geosam2'},  # New
            {'label': 'SAM GEO HQ', 'value': 'samgeo_hq'}  # New
        ],
        value='sam',
        labelStyle={'display': 'inline-block', 'margin': '10px'}
    ),
    html.Button("Generate Mask", id='generate-button', n_clicks=0),
    html.Div(id='image-container', children=[
        html.Img(id='display-image', style={'width': '100%'})
    ]),
    html.Div(id='output-text')
])

# Utility to convert OpenCV image to base64 for displaying in Dash
def convert_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{encoded_image}"

# Utility to draw polygons on an image
def draw_polygons(image, polygons, color=(0, 255, 0)):
    for poly in polygons:
        if isinstance(poly, Polygon):
            coords = np.array(list(poly.exterior.coords), dtype=np.int32)
            cv2.polylines(image, [coords], isClosed=True, color=color, thickness=2)
        elif isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                if sub_poly.is_valid and not sub_poly.is_empty:
                    coords = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                    cv2.polylines(image, [coords], isClosed=True, color=color, thickness=2)
    return image

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

def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
    for existing_poly in existing_polys:
        intersection_area = poly.intersection(existing_poly).area
        if poly.area == 0:
            continue
        overlap_ratio = intersection_area / poly.area
        if overlap_ratio >= area_overlap_threshold:
            return True
    return False

# Define function to check if one mask is completely inside another
def is_mask_inside(outer_mask, inner_mask):
    # Check if all True pixels in inner_mask are also True in outer_mask
    return np.all(outer_mask[inner_mask > 0])

def filter_sam_results(sam_result):
    masks_with_areas_and_bboxes = []
    for i, mask in enumerate(sam_result):
        segmentation = mask['segmentation']
        if np.any(segmentation):
            area = np.sum(segmentation)
            # Compute bounding box
            coords = np.argwhere(segmentation)
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
            bbox = (min_x, min_y, max_x, max_y)
            masks_with_areas_and_bboxes.append((i, segmentation, area, bbox))
    
    # Sort masks by area (from largest to smallest)
    masks_with_areas_and_bboxes.sort(key=lambda x: x[2], reverse=True)  # (index, mask, area, bbox)
    
    # Set the threshold for the minimum number of contained masks required to remove a mask
    contained_mask_threshold = int(0.5 * len(masks_with_areas_and_bboxes))
    
    # Identify masks to remove
    indices_to_remove = set()
    
    # Loop through masks and remove larger masks that contain multiple smaller masks
    for i, (outer_idx, outer_mask, outer_area, outer_bbox) in enumerate(masks_with_areas_and_bboxes):
        outer_min_x, outer_min_y, outer_max_x, outer_max_y = outer_bbox
        contained_count = 0  # Counter for masks contained within the current outer mask
    
        # Only consider smaller masks
        for inner_idx, inner_mask, inner_area, inner_bbox in masks_with_areas_and_bboxes[i+1:]:
            inner_min_x, inner_min_y, inner_max_x, inner_max_y = inner_bbox
    
            # Check if inner bounding box is entirely within outer bounding box
            if (inner_min_x >= outer_min_x and inner_max_x <= outer_max_x and
                inner_min_y >= outer_min_y and inner_max_y <= outer_max_y):
                # Now check if inner_mask is inside outer_mask
                if is_mask_inside(outer_mask, inner_mask):
                    contained_count += 1  # Increment count for each contained mask
    
        # Only mark the larger mask for removal if it contains at least `contained_mask_threshold` smaller masks
        if contained_count >= contained_mask_threshold:
            indices_to_remove.add(outer_idx)
    
    # Filter out the unwanted masks
    filtered_masks_with_areas_and_bboxes = [
        (idx, mask, area, bbox)
        for idx, mask, area, bbox in masks_with_areas_and_bboxes
        if idx not in indices_to_remove
    ]
    
    # Also remove any masks that cover the entire image (if any)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    filtered_masks_with_areas_and_bboxes = [
        (idx, mask, area, bbox)
        for idx, mask, area, bbox in filtered_masks_with_areas_and_bboxes
        if area < image_area
    ]
    
    # Create a filtered sam_result
    filtered_sam_result = [sam_result[idx] for idx, _, _, _ in filtered_masks_with_areas_and_bboxes]

    return filtered_sam_result

@app.callback(
    Output('display-image', 'src'),
    Output('output-text', 'children'),
    Input('generate-button', 'n_clicks'),
    State('mask-generator', 'value')
)

def generate_mask(n_clicks, mask_generator_choice):
    if n_clicks == 0:
        # Show the original image
        encoded_image = convert_image_to_base64(image_rgb)
        return encoded_image, "Select a mask generator and click 'Generate Mask'."

    # Generate masks
    if mask_generator_choice == 'sam':
        CHECKPOINT_PATH = "weights/sam_vit_l_0b3195.pth"
        MODEL_TYPE = "vit_l"
        # Initialize SAM (replace with your SAM implementation)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[MODEL_TYPE](CHECKPOINT_PATH).to("cuda")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,                 # Controls the sampling density
            points_per_batch=32,
            pred_iou_thresh=0.9,                # Increase to filter out low-quality masks
            stability_score_thresh=0.95,        # Increase to keep only stable masks
            stability_score_offset=1.0,         # Adjust for stability calculations
            box_nms_thresh=0.1,                 # Decrease to reduce overlapping masks
            crop_n_layers=1,                    # Reduce complexity
            crop_nms_thresh=0.5,                # Adjust NMS threshold for crops
            min_mask_region_area=5000,          # Increase to filter out small masks (in pixels)
            output_mode="binary_mask"
        )
        sam_result = mask_generator.generate(image_rgb)
    elif mask_generator_choice == 'sam2':
        # Initialize SAM2 (replace with your SAM2 implementation)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths to configuration and checkpoint files
        config_file_path = "./sam2.1_hiera_l.yaml"  # Config file in root directory
        checkpoint_file_path = "./sam2.1_hiera_large.pt"  # Checkpoint file in root directory
        
        # Verify file existence
        assert os.path.exists(config_file_path), f"Config file not found at {config_file_path}"
        assert os.path.exists(checkpoint_file_path), f"Checkpoint file not found at {checkpoint_file_path}"
        
        # Clear any existing Hydra instances
        GlobalHydra.instance().clear()
        
        # Initialize Hydra
        with initialize(config_path="."):
            # Now you can call build_sam2
            print("Attempting to initialize the SAM2 model...")
            sam2_model = build_sam2(config_file=config_file_path, ckpt_path=checkpoint_file_path).to(device)
        
        sam2_model.to(device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=84,
            points_per_batch=16,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            stability_score_offset=1.0,
            mask_threshold=0.0,
            box_nms_thresh=1,           
            crop_n_layers=1,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=0.2,        
            crop_n_points_downscale_factor=1,
            point_grids=None,
            min_mask_region_area=100,
            output_mode="binary_mask",
            use_m2m=True,                  
            multimask_output=False
        )
        sam_result = mask_generator.generate(image_rgb)

    elif mask_generator_choice == 'geosam':
        sam_kwargs = {
            "points_per_side": 32,
            "points_per_batch": 16,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.85,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 100,
        }
        mask_generator = SamGeo(
            model_type="vit_h",
            sam_kwargs=sam_kwargs,
        )
        mask_generator.generate(image_rgb, output="masks2.tif", foreground=True)
        sam_result = mask_generator.masks
        
    elif mask_generator_choice == 'geosam2':
        # Initialize GEOSAM
        mask_generator = SamGeo2(
            model_id="sam2-hiera-large",        # Best for large, detailed tasks.
            automatic=True,                    # Enable automatic mask generation.
            apply_postprocessing=True,         # Clean up masks for smoother edges.
            points_per_side=64,               # Denser grid to ensure detailed masks.
            points_per_batch=32,               # Higher batch size for efficiency.
            pred_iou_thresh=0,               # Medium IOU threshold to merge similar regions.
            stability_score_thresh=0.5,       # Lower stability threshold to allow more masks.
            stability_score_offset=1.0,        # Standard offset for stability.
            mask_threshold=0.0,                # Threshold for binarizing mask logits.
            box_nms_thresh=0.2,                # Reduce overlap for distinct regions.
            crop_n_layers=0,                   # Single crop layer for large maps.
            crop_nms_thresh=0.2,               # Ensure non-maximal suppression between crops.
            crop_overlap_ratio=0.0,            # Overlap between crops to catch edges.
            crop_n_points_downscale_factor=1,  # No downscaling for consistent mask detail.
            min_mask_region_area=100,          # Remove very small noise regions.
            output_mode="binary_mask",         # Binary masks are efficient for visualization.
            use_m2m=True,                      # Enable refinement for cleaner masks.
            multimask_output=False,             # Output multiple masks per region for robustness.
            max_hole_area=0.0,                  # Fill small holes within masks.
            max_sprinkle_area=0.0               # Remove small noise (sprinkles) in masks.
        )
        mask_generator.generate(image_rgb)
        sam_result = mask_generator.masks
    elif mask_generator_choice == 'samgeo_hq':
        sam_kwargs = {
            "points_per_side": 32,
            "points_per_batch": 16,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100,
        }
        mask_generator = SamGeoHQ(
            model_type="vit_h",
            sam_kwargs=sam_kwargs,
        )
        mask_generator.generate(image_rgb)
        sam_result = mask_generator.masks
        
    else:
        return dash.no_update, "Invalid mask generator selected."


    polygons_list = []

    image_with_polygons = image_bgr.copy()

    # Image area
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    
    # List to store polygons with their area
    mask_polygons = []

    filtered_sam_result = filter_sam_results(sam_result)

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

    # Process each polygon
    for idx, poly_dict in enumerate(mask_polygons):
        poly = poly_dict['polygon']
        if not is_polygon_mostly_within(poly, [d['polygon'] for d in filtered_polygons], area_overlap_threshold=0.95):
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

    encoded_image = convert_image_to_base64(image_with_polygons)

    return encoded_image, f"Generated {len(filtered_polygons)} polygons using {mask_generator_choice.upper()}."


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8050)
