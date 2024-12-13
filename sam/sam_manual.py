import streamlit as st
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

# Set Streamlit layout to "wide" to see the entire image
st.set_page_config(layout="wide")

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load SAM model
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to save masks
def save_masks_to_disk():
    # Define base directories
    output_dir = "exported_masks"
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Save the original image
    image_output_path = os.path.join(images_dir, "image.jpg")
    cv2.imwrite(image_output_path, map_image)

    # Check if any masks are stored
    if not st.session_state["masks_list"]:
        st.error("No masks to save. Please annotate the image before exporting.")
        return

    # Initialize an empty mask for the entire image
    height, width = map_image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Define label mapping starting from 1
    label_mapping = {label: idx+1 for idx, label in enumerate(labels)}
    st.write(f"Label mapping: {label_mapping}")

    # Iterate over masks and combine them
    for idx, mask_info in enumerate(st.session_state["masks_list"]):
        mask = mask_info["mask"]
        label = mask_info["label"]
        class_id = label_mapping[label]

        mask_sum = np.sum(mask)
        st.write(f"Mask {idx} for label '{label}' has sum: {mask_sum}")

        combined_mask[mask > 0] = class_id

    # Save the combined mask
    mask_output_path = os.path.join(masks_dir, "mask.png")
    cv2.imwrite(mask_output_path, combined_mask)

    # Save a scaled version of the combined mask for visualization
    max_class_id = max(label_mapping.values())
    scaling_factor = 255 // max_class_id
    scaled_mask = (combined_mask * scaling_factor).astype(np.uint8)
    scaled_mask_output_path = os.path.join(masks_dir, "mask_visualization.png")
    cv2.imwrite(scaled_mask_output_path, scaled_mask)

    # Apply a color map for better visualization
    color_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_JET)
    color_mask_output_path = os.path.join(masks_dir, "mask_color.png")
    cv2.imwrite(color_mask_output_path, color_mask)

    # Optionally, save the annotated image
    annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(annotated_image_path, st.session_state["annotated_image"])


# Load the original map image without scaling
map_image_path = "dataset/aalesund/FOKUS/1504201/201.jpg"
map_image = cv2.imread(map_image_path)

# Initialize session state for masks and annotated image if not set
if "masks_list" not in st.session_state:
    st.session_state["masks_list"] = []
if "annotated_image" not in st.session_state:
    st.session_state["annotated_image"] = map_image.copy()

# Add label selection in the sidebar
labels = ["Residential Area", "Forest", "Shooting Range"]
selected_label = st.sidebar.selectbox("Select Label for Next Mask", labels)

# Button to reset masks
if st.button("Reset Masks"):
    st.session_state["masks_list"].clear()
    st.session_state["annotated_image"] = map_image.copy()

# Button to export masks
if st.button("Export Masks"):
    save_masks_to_disk()
    st.success("Masks have been exported successfully.")

# Display the image on canvas if loaded
if map_image is not None:
    # Convert map_image to PIL format for st_canvas
    pil_image = Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))

    # Display the image with a canvas to capture click coordinates
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

    # Check if a point was clicked
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        # Get the last clicked point
        x_coord = int(canvas_result.json_data["objects"][-1]["left"])
        y_coord = int(canvas_result.json_data["objects"][-1]["top"])
        st.write(f"Clicked coordinates: ({x_coord}, {y_coord})")

        # Define input point for SAM
        input_point = np.array([[x_coord, y_coord]])
        input_label = np.array([1])  # foreground label for SAM

        # Use SamPredictor to set the image and predict the mask
        predictor = SamPredictor(model)
        predictor.set_image(map_image)

        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )

        # Use the first mask
        mask = masks[0]

        # The mask is already at the full image size, no need to resize since no scaling was applied
        mask_resized = (mask > 0).astype(np.uint8)

        # Convert mask to uint8
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # Find contours and draw polygon lines
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cnt.shape[0] >= 3:
                cv2.polylines(st.session_state["annotated_image"], [cnt], True, (0, 255, 0), 5)

        # Store the mask with its label
        st.session_state["masks_list"].append({
            "mask": mask_resized,
            "label": selected_label
        })

    # Display the updated annotated image
    st.image(st.session_state["annotated_image"], channels="BGR")
else:
    st.error("Could not load the image. Please check the file path.")
