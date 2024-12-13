import streamlit as st
import cv2
import numpy as np
import torch
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
from shapely.geometry import MultiPolygon, Polygon

from sam2.sam2_image_predictor import SAM2ImagePredictor

st.set_page_config(layout="wide")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

def save_masks_to_disk():
    output_dir = "exported_masks"
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    image_output_path = os.path.join(images_dir, "image.jpg")
    cv2.imwrite(image_output_path, map_image)

    if not st.session_state["masks_list"]:
        st.error("No masks to save. Please annotate the image before exporting.")
        return

    height, width = map_image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    label_mapping = {label: idx+1 for idx, label in enumerate(labels)}
    st.write(f"Label mapping: {label_mapping}")

    for idx, mask_info in enumerate(st.session_state["masks_list"]):
        mask = mask_info["mask"]
        label = mask_info["label"]
        class_id = label_mapping[label]
        mask_sum = np.sum(mask)
        st.write(f"Mask {idx} for label '{label}' has sum: {mask_sum}")
        combined_mask[mask > 0] = class_id

    mask_output_path = os.path.join(masks_dir, "mask.png")
    cv2.imwrite(mask_output_path, combined_mask)

    max_class_id = max(label_mapping.values())
    scaling_factor = 255 // max_class_id
    scaled_mask = (combined_mask * scaling_factor).astype(np.uint8)
    scaled_mask_output_path = os.path.join(masks_dir, "mask_visualization.png")
    cv2.imwrite(scaled_mask_output_path, scaled_mask)

    color_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_JET)
    color_mask_output_path = os.path.join(masks_dir, "mask_color.png")
    cv2.imwrite(color_mask_output_path, color_mask)

    annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(annotated_image_path, st.session_state["annotated_image"])


# Contour smoothing function
def smooth_contour(contour, window_size=5):
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    contour = np.concatenate((contour[-half_window:], contour, contour[:half_window]), axis=0)
    smoothed_contour = []
    for i in range(half_window, len(contour) - half_window):
        window_points = contour[i - half_window:i + half_window + 1]
        mean_point = np.mean(window_points, axis=0)
        smoothed_contour.append(mean_point)
    smoothed_contour = np.array(smoothed_contour, dtype=np.int32)
    return smoothed_contour

map_image_path = "dataset/aalesund/FOKUS/1504201/201.jpg"
map_image = cv2.imread(map_image_path)
scale_percent = 10
width = int(map_image.shape[1] * scale_percent / 100)
height = int(map_image.shape[0] * scale_percent / 100)
map_image = cv2.resize(map_image, (width, height), interpolation=cv2.INTER_AREA)

if "masks_list" not in st.session_state:
    st.session_state["masks_list"] = []
if "annotated_image" not in st.session_state:
    st.session_state["annotated_image"] = map_image.copy()

labels = ["Residential Area", "Forest", "Shooting Range"]
selected_label = st.sidebar.selectbox("Select Label for Next Mask", labels)

if st.button("Reset Masks"):
    st.session_state["masks_list"].clear()
    st.session_state["annotated_image"] = map_image.copy()

if st.button("Export Masks"):
    save_masks_to_disk()
    st.success("Masks have been exported successfully.")

if map_image is not None:
    pil_image = Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000",
        background_image=pil_image,
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="point",
        key="canvas"
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        x_coord = int(canvas_result.json_data["objects"][-1]["left"])
        y_coord = int(canvas_result.json_data["objects"][-1]["top"])
        st.write(f"Clicked coordinates: ({x_coord}, {y_coord})")

        input_point = np.array([[x_coord, y_coord]])
        input_label = np.array([1])

        model.set_image(map_image)
        with torch.no_grad():
            masks, scores, logits = model.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )

        mask = masks[0]
        original_h, original_w = map_image.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8)

        # Instead of coloring the mask, we now convert it to polygons and draw their outlines.
        image_area = original_h * original_w

        # Find contours
        contours, hierarchy = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Smooth and convert contours to polygons
        mask_polygons = []
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

        # Filter out overly large polygons
        max_area_threshold = 0.1 * image_area
        mask_polygons = [mp for mp in mask_polygons if mp['area'] < max_area_threshold]

        mask_polygons.sort(key=lambda x: x['area'], reverse=True)

        def is_polygon_mostly_within(poly, existing_polys, area_overlap_threshold=0.95):
            for existing_poly in existing_polys:
                intersection_area = poly.intersection(existing_poly).area
                if poly.area == 0:
                    continue
                overlap_ratio = intersection_area / poly.area
                if overlap_ratio >= area_overlap_threshold:
                    return True
            return False

        filtered_polygons = []
        for poly_dict in mask_polygons:
            poly = poly_dict['polygon']
            if not is_polygon_mostly_within(poly, [d['polygon'] for d in filtered_polygons], area_overlap_threshold=0.05):
                filtered_polygons.append(poly_dict)

        # Draw polygons on the annotated image
        annotated_image = st.session_state["annotated_image"].copy()
        for poly_dict in filtered_polygons:
            poly = poly_dict['polygon']
            if isinstance(poly, Polygon):
                coords = np.array(list(poly.exterior.coords)).astype(np.int32)
                cv2.polylines(annotated_image, [coords], isClosed=True, color=(0, 255, 0), thickness=1)
            elif isinstance(poly, MultiPolygon):
                for sub_poly in poly.geoms:
                    if sub_poly.is_valid and not sub_poly.is_empty:
                        coords = np.array(list(sub_poly.exterior.coords)).astype(np.int32)
                        cv2.polylines(annotated_image, [coords], isClosed=True, color=(0, 255, 0), thickness=1)

        # Update session state
        st.session_state["annotated_image"] = annotated_image
        # Store the mask and label as before
        st.session_state["masks_list"].append({
            "mask": mask_resized,
            "label": selected_label
        })

    st.image(st.session_state["annotated_image"], channels="BGR")
else:
    st.error("Could not load the image. Please check the file path.")
