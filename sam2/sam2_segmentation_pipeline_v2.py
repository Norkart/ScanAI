import os
import re
import glob
import cv2
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon, mapping
import geojson
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ------------------------------------------------------------------------------
# 1) Helper to load either .jpg + .jgw OR .tif(EPSG:25832) 
#    and return (image_bgr, (A,B,C,D,E,F)) for pixel->world
# ------------------------------------------------------------------------------
def load_image_and_transform(path):
    """
    Given a path to either:
      - a .jpg (with matching .jgw)
      - or a .tif (with EPSG:25832)
    returns:
      image_bgr      : a (H, W, 3) numpy array in BGR order
      (A,B,C,D,E,F)  : transformation parameters for pixel->world
    If there's any failure or mismatch, return None, None.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jpg":
        # Expect a matching .jgw
        jgw_path = os.path.splitext(path)[0] + ".jgw"
        if not os.path.exists(jgw_path):
            print(f"No .jgw file for {path}, skipping.")
            return None, None
        
        # 1) Load the .jpg
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"Could not read {path}, skipping.")
            return None, None
        
        # 2) Parse the .jgw
        try:
            A, B, C, D, E, F = read_jgw(jgw_path)
        except Exception as e:
            print(f"Could not parse {jgw_path}: {e}, skipping.")
            return None, None

        return image_bgr, (A,B,C,D,E,F)
    
    elif ext == ".tif":
        # Attempt to open with rasterio, check if EPSG:25832
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None or str(ds.crs) != "EPSG:25832":
                    print(f"{path} is not EPSG:25832, skipping.")
                    return None, None
                
                # read first up to 3 bands
                data = ds.read()
                band_count = min(data.shape[0], 3)
                # shape = (count, height, width)
                arr = np.transpose(data[:band_count, :, :], (1, 2, 0)).astype(np.uint8)

                # If 1-band, replicate it
                if band_count == 1:
                    arr = cv2.merge([arr[:,:,0], arr[:,:,0], arr[:,:,0]])
                elif band_count == 2:
                    # not typical, but replicate the second band to form 3
                    arr = cv2.merge([arr[:,:,0], arr[:,:,1], arr[:,:,1]])
                # If 3, we assume it's RGB
                if band_count >= 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                image_bgr = arr

                # Now extract transform 
                affine: Affine = ds.transform
                # Rasterio's transform is:  (a, b, c, d, e, f)
                # Where x = a*col + b*row + c,  y = d*col + e*row + f
                # We'll rename them A,B,C,D,E,F consistently
                a = affine.a
                b = affine.b
                c = affine.xoff
                d = affine.d
                e = affine.e
                f = affine.yoff
                A,B,C,D,E,F = a,b,c,d,e,f

                return image_bgr, (A,B,C,D,E,F)

        except Exception as e:
            print(f"Could not open .tif {path} with rasterio: {e}")
            return None, None
    else:
        print(f"Unsupported extension for {path}, skipping.")
        return None, None


def read_jgw(jgw_file):
    """
    Reads the 6 lines of a .jgw file into their transform variables:
    A, B, C, D, E, F
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
    Convert pixel coords (px, py) to real-world coords using .jgw transform style:
      X = A*px + B*py + C
      Y = D*px + E*py + F
    """
    X = A * px + B * py + C
    Y = D * px + E * py + F
    return (X, Y)


# ------------------------------------------------------------------------------
# 2) Single function to do segmentation + filtering + drawing + geojson
# ------------------------------------------------------------------------------
def segment_and_generate_geojson_unified(
    image_bgr,
    transform_params,
    input_basename,
    mask_generator,
    output_folder
):
    """
    Given:
      image_bgr       : the loaded image in BGR (height x width x 3)
      transform_params: (A,B,C,D,E,F) for pixel->world
      input_basename  : the base name of the input file (without extension)
      mask_generator  : the SAM2 mask generator
      output_folder   : where to save .geojson and _drawn.jpg
    
    Produces:
      - <output_folder>/<input_basename>.geojson
      - <output_folder>/<input_basename>_drawn.jpg
    """
    A,B,C,D,E,F = transform_params
    
    # Convert to RGB for SAM
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step B: Generate raw masks
    sam_result = mask_generator.generate(image_rgb)
    print(f"{input_basename}: total raw masks = {len(sam_result)}")

    # Step C: Filtering logic
    from shapely.geometry import Polygon
    def is_mask_inside(outer_mask, inner_mask):
        return np.all(outer_mask[inner_mask > 0])

    def is_colorful_region(image, mask, saturation_threshold=20, brightness_threshold=50):
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
    
    # Build a list of (index, segmentation, area, bbox)
    masks_with_areas_and_bboxes = []
    for i, mask_dict in enumerate(sam_result):
        seg = mask_dict['segmentation']
        if np.any(seg):
            area = np.sum(seg)
            coords = np.argwhere(seg)
            y_coords, x_coords = coords[:,0], coords[:,1]
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
            bbox = (min_x, min_y, max_x, max_y)
            masks_with_areas_and_bboxes.append((i, seg, area, bbox))

    # Sort by area (descending)
    masks_with_areas_and_bboxes.sort(key=lambda x: x[2], reverse=True)

    contained_mask_threshold = int(0.1 * len(masks_with_areas_and_bboxes))
    indices_to_remove = set()

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

    filtered_masks = [
        (idx, seg, area, bbox)
        for idx, seg, area, bbox in masks_with_areas_and_bboxes
        if idx not in indices_to_remove
    ]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    filtered_masks = [
        (idx, seg, area, bbox)
        for idx, seg, area, bbox in filtered_masks
        if area < image_area
    ]

    filtered_masks = [
        (idx, seg, area, bbox)
        for idx, seg, area, bbox in filtered_masks
        if is_colorful_region(image_bgr, seg, saturation_threshold=0, brightness_threshold=0)
    ]

    filtered_sam_result = [sam_result[idx] for idx,_,_,_ in filtered_masks]
    print(f"{input_basename}: total masks after filtering = {len(filtered_sam_result)}")

    # Step D: Convert masks->Contours->Polygons->Filter
    def smooth_contour(contour, window_size=15):
        if window_size % 2 == 0:
            window_size += 1
        half_window = window_size // 2
        contour = np.concatenate((contour[-half_window:], contour, contour[:half_window]), axis=0)
        smoothed = []
        for i in range(half_window, len(contour) - half_window):
            window_points = contour[i - half_window:i + half_window + 1]
            mean_point = np.mean(window_points, axis=0)
            smoothed.append(mean_point)
        return np.array(smoothed, dtype=np.int32)

    from shapely.geometry import Polygon, MultiPolygon
    mask_polygons = []
    for idx, dict_ in enumerate(filtered_sam_result):
        seg = dict_['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        for contour in contours:
            if contour.shape[0] < 5:
                continue
            contour = contour.reshape(-1,2)
            sc = smooth_contour(contour, 15)
            if sc.shape[0] >=3:
                poly = Polygon(sc)
                if not poly.is_valid or poly.area == 0:
                    poly = poly.buffer(0)
                    if not poly.is_valid or poly.area == 0:
                        continue
                mask_polygons.append({'area': poly.area, 'polygon': poly})

    # Exclude overly large polygons
    max_area_threshold = 0.1 * image_area
    mask_polygons = [p for p in mask_polygons if p['area'] < max_area_threshold]
    print(f"{input_basename}: polygons after excluding large masks = {len(mask_polygons)}")

    mask_polygons.sort(key=lambda x: x['area'], reverse=True)
    filtered_polygons = []

    def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
        for ep in existing_polys:
            inter_area = poly.intersection(ep).area
            if poly.area == 0:
                continue
            if (inter_area / poly.area) >= area_overlap_threshold:
                return True
        return False

    for p in mask_polygons:
        poly = p['polygon']
        if not is_polygon_mostly_within(poly, [fp['polygon'] for fp in filtered_polygons], 0.05):
            filtered_polygons.append(p)

    print(f"{input_basename}: polygons after final overlap filtering = {len(filtered_polygons)}")

    # Draw final polygons on the BGR image
    for p in filtered_polygons:
        poly = p['polygon']
        if isinstance(poly, Polygon):
            coords = np.array(list(poly.exterior.coords)).astype(np.int32)
            cv2.polylines(image_bgr, [coords], True, (0,255,0), thickness=3)
        elif isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                coords = np.array(list(sub_poly.exterior.coords)).astype(np.int32)
                cv2.polylines(image_bgr, [coords], True, (0,255,0), thickness=3)

    # Save the drawn image
    out_drawn_path = os.path.join(output_folder, f"{input_basename}_drawn.jpg")
    cv2.imwrite(out_drawn_path, image_bgr)
    print(f"Saved mask outlines to: {out_drawn_path}")

    # Convert to world coords, save GeoJSON
    def polygon_to_world_coords(poly, A,B,C,D,E,F):
        exterior = []
        for px, py in poly.exterior.coords:
            xw, yw = pixel_to_world(px, py, A,B,C,D,E,F)
            exterior.append((xw,yw))
        interiors = []
        for interior in poly.interiors:
            hole = []
            for px, py in interior.coords:
                xw, yw = pixel_to_world(px, py, A,B,C,D,E,F)
                hole.append((xw,yw))
            interiors.append(hole)
        return Polygon(exterior, interiors)
    
    wpolys = []
    for p in filtered_polygons:
        poly = p['polygon']
        if isinstance(poly, Polygon):
            w_poly = polygon_to_world_coords(poly, A,B,C,D,E,F)
            if w_poly.is_valid and not w_poly.is_empty:
                wpolys.append(w_poly)
        else:
            for sub_poly in poly.geoms:
                w_poly = polygon_to_world_coords(sub_poly, A,B,C,D,E,F)
                if w_poly.is_valid and not w_poly.is_empty:
                    wpolys.append(w_poly)

    feats = []
    for wpoly in wpolys:
        feats.append(geojson.Feature(geometry=mapping(wpoly), properties={}))
    fc = geojson.FeatureCollection(feats)
    out_geojson_path = os.path.join(output_folder, f"{input_basename}.geojson")
    with open(out_geojson_path, 'w') as f:
        geojson.dump(fc, f, indent=2)
    print(f"Saved {len(wpolys)} polygons to {out_geojson_path}")


# ------------------------------------------------------------------------------
# 3) MAIN: Find all .jpg + .jgw or .tif (EPSG:25832), run everything once
# ------------------------------------------------------------------------------
def find_all_supported_images(root_dir):
    """
    Return a list of absolute paths for:
      - .jpg files that have a matching .jgw
      - .tif files (we'll check EPSG:25832 inside load_image_and_transform)
    Sorted in alphabetical order
    """
    results = []
    for subdir, dirs, files in os.walk(root_dir):
        files.sort()
        for f in files:
            ext = f.lower()
            if ext.endswith('.jpg'):
                # We only want it if there's a .jgw
                jgw_candidate = os.path.join(subdir, os.path.splitext(f)[0] + '.jgw')
                if os.path.exists(jgw_candidate):
                    full_path = os.path.join(subdir, f)
                    results.append(full_path)
            elif ext.endswith('.tif'):
                # We'll let load_image_and_transform decide if it's EPSG:25832
                full_path = os.path.join(subdir, f)
                results.append(full_path)
    # Sort
    results.sort()
    return results


if __name__ == "__main__":
    # Use bfloat16 for the entire script
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id="facebook/sam2-hiera-large",
        points_per_side=32,
        points_per_batch=8,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
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
    print("Model initialized successfully.")

    root_dir = "dataset_with_jgw"
    output_folder = "geojson_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Gather all possible images in a single list
    all_paths = find_all_supported_images(root_dir)

    # For each, load image + transform, then run unified segmentation
    for path in all_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[Processing] {path}")
        image_bgr, transform_params = load_image_and_transform(path)
        if image_bgr is None or transform_params is None:
            continue  # skip
        segment_and_generate_geojson_unified(
            image_bgr, 
            transform_params,
            base_name,
            mask_generator,
            output_folder
        )

    print("All done!")
