import os
import glob
import cv2
import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon, mapping
import geojson
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ---------------------------------------------------------------------
# 1) Helpers to traverse directories & read .jgw transforms
# ---------------------------------------------------------------------

def find_jpg_jgw_pairs(root_dir):
    """
    Recursively traverse root_dir, collect (jpg_path, jgw_path) pairs, 
    then return them in sorted order by jpg_path.
    """
    pairs = []
    for subdir, dirs, files in os.walk(root_dir):
        # Sort file list to ensure a consistent traversal order
        files.sort()
        jpg_files = [f for f in files if f.lower().endswith('.jpg')]
        for f in jpg_files:
            jpg_path = os.path.join(subdir, f)
            base_name = os.path.splitext(jpg_path)[0]
            jgw_path = base_name + '.jgw'
            if os.path.exists(jgw_path):
                pairs.append((jpg_path, jgw_path))
    # Sort pairs by the jpg_path (alphabetically)
    pairs.sort(key=lambda p: p[0])
    return pairs


def read_jgw(jgw_file):
    """
    Reads the 6 lines of a .jgw file into their corresponding transform variables:
    
    A, D, B, E, C, F
    """
    with open(jgw_file, 'r') as f:
        lines = f.read().splitlines()
    A = float(lines[0])  # pixel size in x-direction
    D = float(lines[1])  # rotation term
    B = float(lines[2])  # rotation term
    E = float(lines[3])  # pixel size in y-direction
    C = float(lines[4])  # x-coordinate of center of top-left pixel
    F = float(lines[5])  # y-coordinate of center of top-left pixel
    return A, B, C, D, E, F


def pixel_to_world(px, py, A, B, C, D, E, F):
    """
    Convert pixel coords (px, py) to real-world coords using .jgw transform.
    
    X = A*px + B*py + C
    Y = D*px + E*py + F
    """
    X = A * px + B * py + C
    Y = D * px + E * py + F
    return (X, Y)

# ---------------------------------------------------------------------
# 2) Your existing segmentation + filtering pipeline
#    (wrapped in a function for a single image)
# ---------------------------------------------------------------------

def is_mask_inside(outer_mask, inner_mask):
    """
    Check if all True pixels in inner_mask are also True in outer_mask.
    """
    return np.all(outer_mask[inner_mask > 0])


def is_colorful_region(image, mask, saturation_threshold=20, brightness_threshold=50):
    """
    Checks if the region within the mask is sufficiently colorful and bright.
    """
    if mask.ndim == 2:
        mask = mask.astype(np.uint8)
    else:
        raise ValueError("The mask should be a 2D binary array.")

    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_scaled = mask * 255

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_scaled)

    saturation = masked_hsv[..., 1]
    brightness = masked_hsv[..., 2]

    mean_saturation = cv2.mean(saturation, mask=mask_scaled)[0]
    mean_brightness = cv2.mean(brightness, mask=mask_scaled)[0]

    return mean_saturation > saturation_threshold and mean_brightness > brightness_threshold


def smooth_contour(contour, window_size=5):
    """
    Smooth contour points using a moving average with the specified window_size.
    """
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


def segment_and_generate_geojson(jpg_path, jgw_path, mask_generator, output_geojson_path, output_drawn_jpg_path):
    """
    1) Load .jpg and read .jgw
    2) Generate masks w/ SAM2 + filtering
    3) Draw polygon outlines on the original image, save as a .jpg
    4) Convert final polygon coordinates from pixel -> world, save as GeoJSON
    """

    # -----------------------------------------------------------------
    # Step A: Load image and read world-file transform
    # -----------------------------------------------------------------
    image_bgr = cv2.imread(jpg_path)
    if image_bgr is None:
        print(f"Could not read {jpg_path}, skipping.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        A, B, C, D, E, F = read_jgw(jgw_path)
    except:
        print(f"Could not parse {jgw_path}, skipping.")
        return

    # -----------------------------------------------------------------
    # Step B: Generate raw masks w/ SAM2
    # -----------------------------------------------------------------
    sam_result = mask_generator.generate(image_rgb)
    print(f"{jpg_path}: total raw masks = {len(sam_result)}")

    # -----------------------------------------------------------------
    # Step C: Filtering logic (area, containment, color, etc.)
    # -----------------------------------------------------------------

    # 1) Build a list of (index, segmentation, area, bbox)
    masks_with_areas_and_bboxes = []
    for i, mask in enumerate(sam_result):
        segmentation = mask['segmentation']
        if np.any(segmentation):
            area = np.sum(segmentation)
            coords = np.argwhere(segmentation)
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
            bbox = (min_x, min_y, max_x, max_y)
            masks_with_areas_and_bboxes.append((i, segmentation, area, bbox))

    # Sort by area (descending)
    masks_with_areas_and_bboxes.sort(key=lambda x: x[2], reverse=True)

    # Threshold for removing large masks containing many smaller masks
    contained_mask_threshold = int(0.1 * len(masks_with_areas_and_bboxes))
    indices_to_remove = set()

    # 2) Remove large masks containing multiple smaller ones
    for i, (outer_idx, outer_mask, outer_area, outer_bbox) in enumerate(masks_with_areas_and_bboxes):
        outer_min_x, outer_min_y, outer_max_x, outer_max_y = outer_bbox
        contained_count = 0
        for inner_idx, inner_mask, inner_area, inner_bbox in masks_with_areas_and_bboxes[i+1:]:
            inner_min_x, inner_min_y, inner_max_x, inner_max_y = inner_bbox
            if (inner_min_x >= outer_min_x and inner_max_x <= outer_max_x and
                inner_min_y >= outer_min_y and inner_max_y <= outer_max_y):
                if is_mask_inside(outer_mask, inner_mask):
                    contained_count += 1
        if contained_count >= contained_mask_threshold and contained_mask_threshold != 0:
            indices_to_remove.add(outer_idx)

    filtered_masks_with_areas_and_bboxes = [
        (idx, mask, area, bbox)
        for idx, mask, area, bbox in masks_with_areas_and_bboxes
        if idx not in indices_to_remove
    ]

    # 3) Remove masks that cover the entire image
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    filtered_masks_with_areas_and_bboxes = [
        (idx, mask, area, bbox)
        for idx, mask, area, bbox in filtered_masks_with_areas_and_bboxes
        if area < image_area
    ]

    # 4) Remove masks on predominantly gray/black backgrounds
    filtered_masks_with_areas_and_bboxes = [
        (idx, mask, area, bbox)
        for idx, mask, area, bbox in filtered_masks_with_areas_and_bboxes
        if is_colorful_region(image_bgr, mask, saturation_threshold=0, brightness_threshold=0)
    ]

    # Final filtered SAM results
    filtered_sam_result = [sam_result[idx] for idx, _, _, _ in filtered_masks_with_areas_and_bboxes]
    print(f"{jpg_path}: total masks after filtering = {len(filtered_sam_result)}")

    # -----------------------------------------------------------------
    # Step D: Convert masks -> shapely Polygons (pixel coords)
    # -----------------------------------------------------------------
    mask_polygons = []
    for idx, mask_dict in enumerate(filtered_sam_result):
        mask = mask_dict['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        for contour in contours:
            if contour.shape[0] < 5:
                continue
            contour = contour.reshape(-1, 2)
            smoothed_contour = smooth_contour(contour, window_size=15)
            if smoothed_contour.shape[0] >= 3:
                polygon = Polygon(smoothed_contour)
                if not polygon.is_valid or polygon.area == 0:
                    polygon = polygon.buffer(0)
                    if not polygon.is_valid or polygon.area == 0:
                        continue
                mask_polygons.append({'area': polygon.area, 'polygon': polygon})

    # Exclude overly large polygons (> 10% of image)
    max_area_threshold = 0.1 * image_area
    mask_polygons = [mp for mp in mask_polygons if mp['area'] < max_area_threshold]
    print(f"{jpg_path}: polygons after excluding large masks = {len(mask_polygons)}")

    # Sort polygons by area (descending) to remove smaller polygons fully contained in bigger ones
    mask_polygons.sort(key=lambda x: x['area'], reverse=True)
    filtered_polygons = []

    def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
        for existing_poly in existing_polys:
            intersection_area = poly.intersection(existing_poly).area
            if poly.area == 0:
                continue
            overlap_ratio = intersection_area / poly.area
            if overlap_ratio >= area_overlap_threshold:
                return True
        return False

    for idx, poly_dict in enumerate(mask_polygons):
        poly = poly_dict['polygon']
        if not is_polygon_mostly_within(poly, [d['polygon'] for d in filtered_polygons], area_overlap_threshold=0.05):
            filtered_polygons.append(poly_dict)

    print(f"{jpg_path}: polygons after final overlap filtering = {len(filtered_polygons)}")

    # -----------------------------------------------------------------
    # (NEW) Step D2: Draw the final pixel polygons on the image
    # -----------------------------------------------------------------
    # We'll draw them on the BGR copy (the original image). 
    for poly_dict in filtered_polygons:
        poly = poly_dict['polygon']
        if isinstance(poly, Polygon):
            coords = np.array(list(poly.exterior.coords)).astype(np.int32)
            cv2.polylines(image_bgr, [coords], isClosed=True, color=(0, 255, 0), thickness=3)
        elif isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                coords = np.array(list(sub_poly.exterior.coords)).astype(np.int32)
                cv2.polylines(image_bgr, [coords], isClosed=True, color=(0, 255, 0), thickness=3)

    # Now save the "drawn-on" image
    cv2.imwrite(output_drawn_jpg_path, image_bgr)
    print(f"Saved mask outlines image to: {output_drawn_jpg_path}")

    # -----------------------------------------------------------------
    # Step E: Convert pixel polygons -> real-world polygons, save GeoJSON
    # -----------------------------------------------------------------

    def polygon_to_world_coords(polygon, A, B, C, D, E, F):
        exterior_coords = []
        for px, py in polygon.exterior.coords:
            xw, yw = pixel_to_world(px, py, A, B, C, D, E, F)
            exterior_coords.append((xw, yw))
        interiors = []
        for interior in polygon.interiors:
            hole_coords = []
            for px, py in interior.coords:
                xw, yw = pixel_to_world(px, py, A, B, C, D, E, F)
                hole_coords.append((xw, yw))
            interiors.append(hole_coords)
        return Polygon(exterior_coords, interiors)

    world_polygons = []
    for poly_dict in filtered_polygons:
        poly = poly_dict['polygon']
        if isinstance(poly, Polygon):
            w_poly = polygon_to_world_coords(poly, A, B, C, D, E, F)
            if w_poly.is_valid and not w_poly.is_empty:
                world_polygons.append(w_poly)
        elif isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                if sub_poly.is_valid and not sub_poly.is_empty:
                    w_poly = polygon_to_world_coords(sub_poly, A, B, C, D, E, F)
                    if w_poly.is_valid and not w_poly.is_empty:
                        world_polygons.append(w_poly)

    # Write the final polygons to a GEOJSON file
    features = []
    for poly in world_polygons:
        feat = geojson.Feature(geometry=mapping(poly), properties={})
        features.append(feat)
    fc = geojson.FeatureCollection(features)
    
    with open(output_geojson_path, 'w') as f:
        geojson.dump(fc, f, indent=2)

    print(f"Saved {len(world_polygons)} polygons to {output_geojson_path}")


# ---------------------------------------------------------------------
# 3) Main script: set up model, loop over all .jpg/.jgw pairs
#    -> save BOTH GeoJSON and the drawn-on .jpg
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Use bfloat16 for the entire script
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Initialize SAM2 Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id="facebook/sam2-hiera-large",
        points_per_side=32,  # Define points per side
        points_per_batch=16,  # Number of points per batch
        pred_iou_thresh=0.75,  # Filter threshold for mask quality
        stability_score_thresh=0.75,  # Filter threshold for stability score
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.5,           
        crop_n_layers=1,
        crop_nms_thresh=1,
        crop_overlap_ratio=0.8,        
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=0,
        output_mode="binary_mask",
        use_m2m=True,                  
        multimask_output=False
    )
    
    print("Model and mask generator initialized successfully.")

    root_dir = "dataset_with_jgw/aalesund/FOKUS"
    output_folder = "geojson_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Traverse for .jpg/.jgw
    for jpg_path, jgw_path in find_jpg_jgw_pairs(root_dir):
        base_name = os.path.basename(jpg_path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # 1) .geojson file
        output_geojson_path = os.path.join(output_folder, f"{name_no_ext}.geojson")
        # 2) drawn .jpg
        output_drawn_jpg_path = os.path.join(output_folder, f"{name_no_ext}_drawn.jpg")

        print(f"\n[Processing] {jpg_path} with {jgw_path}")
        segment_and_generate_geojson(
            jpg_path, 
            jgw_path, 
            mask_generator, 
            output_geojson_path,
            output_drawn_jpg_path
        )
    
    print("All done!")
