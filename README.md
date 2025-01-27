# SCANAI

**SCANAI** is a project focused on segmenting map images by identifying specific regions and converting them into vectorized polygons. The goal is to automate the segmentation process on hand-marked/hand-annotated map images.

## Project Overview
SCANAI explores various segmentation methods to identify the best approach for converting hand-marked map images into structured, vectorized data.

## Segmentation Methods Explored

### Segment Anything Model's

#### Segment Anything Model (SAM)

- **SAM1**: Provides the best out-of-the-box performance for segmentation tasks.
- **Coordinate Prompting with SAM**: Highest precision but can only process one region at a time.
- *Additional notes:*
    - Reliable out-of-the-box
    - Good for initial tests on map segmentation
    - Easy to set up
- **Things Tried:**
    - Automatic mask generator
    - Parameter tuning
    - **Coordinate Prompting**:
        - Used Streamlit canvas for interactive point selection.
        - Captured click coordinates to input into SAM's predictor.
        - Applied single-point prompts to generate individual mask regions.

#### Segment Anything Model 2 (SAM2)

- **SAM2**: SAM2 outperforms SAM1 when parameters are fine-tuned.
- *Additional notes:*
    - Requires tuning for optimal results
    - Fine-tuning reveals potential for superior segmentation accuracy
    - More difficult to set up than standard SAM
- **Things Tried:**
    - Automatic mask generator
        - **Parameters used for best results**:
            - `points_per_side=64`: Specifies the grid resolution for mask generation; higher values increase precision but require more computation.
            - `points_per_batch=64`: Controls batch processing for points, optimizing resource usage during mask generation.
            - `pred_iou_thresh=0.8`: Sets the IoU threshold for mask prediction confidence; higher values reduce false positives.
            - `stability_score_thresh=0.9`: Filters masks based on stability score; higher values ensure only the most stable masks are kept.
            - `stability_score_offset=1.0`: Adjusts the stability score sensitivity, balancing between stability and coverage.
            - `mask_threshold=0.0`: Defines the threshold for mask pixel inclusion, where 0.0 includes all pixels.
            - `box_nms_thresh=1`: Sets the non-maximum suppression (NMS) threshold for bounding box overlap; here, set to max overlap to retain more masks.
            - `crop_n_layers=1`: Number of layers for cropping, which refines mask boundaries.
            - `crop_nms_thresh=0.7`: Controls NMS for cropped areas, reducing overlapping mask redundancy.
            - `crop_overlap_ratio=0.2`: Determines overlap allowance for cropped segments, balancing coverage.
            - `crop_n_points_downscale_factor=1`: Scales the number of points for cropped areas; 1 keeps original scale.
            - `point_grids=None`: No predefined grid is set, allowing for automatic point grid generation.
            - `min_mask_region_area=0`: No minimum region area for mask filtering, retaining all mask sizes.
            - `output_mode="binary_mask"`: Output format set to binary mask for straightforward region delineation.
            - `use_m2m=True`: Enables mask-to-mask refinement, improving mask accuracy.
            - `multimask_output=False`: Outputs a single mask per region, ensuring only the best mask per prompt.

#### SAMGEO

##### SAMGEO Automatic
- Gives similar results as standard SAM

##### SAMGEO Automatic HQ
- The High Quality SAMGEO model does not seem to give any noticable better performance than standard SAMGEO
- Takes more time to process and segment map images. 

##### SAMGEO2 Automatic
- SAMGEO2 seems to give more accurate masks on certain map areas than SAM2
- Struggles to segment cetrain large areas compared to SAM2
- Still pretty good results and SAMGEO2 and SAM2 seems to be the contenders for the best automatic mask generators

##### SAMGEO Language
- SAMGEO with language prompting does not seem to give any good results on map images. Difficult to find a prompt for finding all the different areas in a map



#### Standard OpenCV Techniques

- **Standard Techniques**: OpenCV was tested for baseline segmentation but yielded less effective results.
- *Additional notes:*
    - Less effective compared to SAM models
    - May not be suitable for complex or detailed map segmentations
    - Maps being handmarked with different markers makes selecting a proper threshold difficult
- **Things Tried:**
    - Thresholding and edge detection
    - Contour detection


## K-Means Image Clustering

Used K-Means clustering with silhouette score to categorize images into meaningful clusters based on visual features. Pre-trained deep learning models were used to extract features from the images, and their performance was evaluated on different clustering configurations.

### Steps
1. Collect images and PDFs recursively; convert PDFs to images.
2. Extract deep features using ResNet50.
3. Reduce features to 50 dimensions with PCA.
4. Find optimal clusters using silhouette scores (2–15 clusters).
5. Cluster images with Agglomerative Clustering and save to directories.

### Results
- **Efficientnet** showed the best performance on 3 clusters, clustering the dataset into **old maps**, **small areas**, and **large colorful maps**.
- **ResNet** showed the best performance on 2 clusters, clustering the dataset into **old uncolored maps** and **new colorful maps**.

The resulting clusters can be seen in [Excel sheet](https://docs.google.com/spreadsheets/d/1tYTSKLr1oZa4xqcuU85iRogTT5oW9OQQVPY2q-Be3T8/edit?gid=1974503951#gid=1974503951).

## Vision Models Explored


### GPT4-Turbo Vision

The GPT4-Turbo Vision model demonstrated mixed results when applied to map-related tasks:

#### Unpromising Results:
- **General OCR on Maps**: The model performed poorly in reading text directly from the maps. When prompted to extract all text from a map it would either say that it cannot do that, or hallucinate and say text that was not in the map at all.
- **Area Detection**: It struggled to identify and distinguish the number of different areas present in the maps. When prompted to count the amount of areas of a certain color in a map it would give a different answer every time it was asked.
- **Coordinate Extraction**: The model was unable to extract geographic coordinates from map images. It does not have the built in functionality to give image coordinates when prompted.

#### Promising Results:
- **Map Explanation/Legend Analysis**: The model showed potential in analyzing legend images, accurately identifying the connection between map area names and their corresponding color explanations. When prompted to give all area names and their associated colors in a map explanation/legend it would get all of them correct most of the time.

Overall, while GPT4-Turbo Vision has significant limitations in core map-reading tasks, it shows promise in legend-based analysis.

### Azure AI Vision

The Azure AI Vision model showed both strengths and limitations when applied to map-related tasks:

#### Promising Results:
- **Text Extraction**: Showed promise in extracting text from both map images and map explanation/legend images.
- **Bounding Boxes**: Provide bounding boxes for the detected text, giving us the location of the text is within the image.

#### Limitations:
- **Small Text**: The model struggles to extract all text from map images, especially where small text is densely packed, limiting its effectiveness in highly detailed maps.
- **Customization**: There is a lack of customization for the model and we cant change any thresholds for text extraction.

Overall, Azure AI Vision is a promising tool for text extraction tasks, particularly when bounding box data is needed, though it may require supplemental methods for handling smaller text.


### YOLO
- Did not work well as YOLO is trained to detect certain objects like book, bench, etc, and does not have functionalities for detecting objects like color boxes in a map explanation. 
- Would need fine tuning to work well and detect map and map explanation related objects.


## Status and Key Findings

- **Out-of-the-Box Performance**: SAM1 generally delivers good results with minimal setup.
- **Parameter Tuning**: SAM2 can outperform SAM1, demonstrating that segmentation quality can improve with parameter adjustments.
- **Coordinate Prompting**: SAM with coordinate prompting offers the best precision but is limited to one region per prompt.
- **OpenCV Performance**: OpenCV's standard techniques are not as effective for this specific task.

## Considerations for "Good Enough" Quality

A critical discussion point is determining what level of segmentation accuracy is "good enough." While automatic segmentation is promising, achieving ideal results may require fine-tuning with a custom, annotated dataset, which could demand significant resources.

# Main Segmentation Pipeline

## Overview

This repository contains a **segmentation pipeline** that automates a series of tasks on a dataset of georeferenced images. The pipeline:

1. **Discovers images** (JPEG, TIFF, PDF, PNG) in a specified dataset directory.
2. **Loads** each image and, if available, **extracts** georeferencing data (e.g., `.jgw` files for `.jpg`, or checking EPSG:25832 for `.tif`).
3. **Runs** the **Segment Anything 2 (SAM2)** model to automatically generate segmentation masks.
4. **Extracts polygons** for each mask (using OpenCV + Shapely), **filters** invalid or uninteresting masks, and **draws** these polygons on the image.
5. **Determines** a “dominant color” for each mask region by **clustering pixel values**, ignoring small variations.
6. **Calls** GPT-4 Vision (using a PDF legend) to guess an “area name” that matches each mask’s color.
7. **Analyzes** each mask region with **Azure Vision**, extracting text found inside it (e.g., via `READ` feature).
8. **Writes** the polygons and associated data (color, area name, recognized text) to a **GeoJSON** file, transforming pixel coordinates to real-world coordinates if available.

## Key Steps

1. **Finding and Loading Images**  
   - Uses `find_all_supported_images(root_dir)` to scan for files in supported formats (`.jpg`, `.tif`, `.pdf`, `.png`).
   - Each discovered file is represented by an `ImageItem` class containing:
     - `path` (the file’s location)
     - `image_type` (e.g. `"jpg_jgw"`, `"jpg_no_jgw"`, `"tif_ok"`, `"tif_bad"`, `"pdf"`, `"png"`)
     - `image_bgr` (the loaded OpenCV image in BGR format)
     - `transform_params` (`(A,B,C,D,E,F)` tuple if georeferencing is known, else `None`)

2. **Segmentation with SAM2**  
   - For each image, the pipeline calls `segment_and_generate_geojson_unified`:
     1. Generates masks via **Segment Anything 2** (SAM2).
     2. Filters large or nested masks.
     3. Converts masks to polygons, draws them in `<basename>_drawn.jpg` for visualization.

3. **Color Classification + GPT-4 Legend Matching**  
   - Clusters pixels in each mask region to find a “dominant color,” skipping grayish pixels if a colorful cluster is present.
   - Calls GPT-4 Vision (given a PDF “tegnforklaring” in the same folder) to guess an **`area_name`** for that color.
   - Note for legend matching to work the dataset needs a map image (i.e. 200.jpg) and a corresponding tegnforklaring image (i.e. 200_tegnforklaring.jpg)

4. **Azure Vision**  
   - Crops each mask region from the original image.
   - Sends the cropped image bytes to **Azure Vision** (`ImageAnalysisClient`) to detect text or objects.
   - Saves recognized text lines in `"text_found"` property.

5. **GeoJSON Creation**  
   - If real georeferencing exists (`transform_params != None`), each polygon is **transformed** from pixel to world coordinates.
   - Otherwise, an identity transform is used (so polygons are stored in pixel space).
   - Outputs to `<basename>_masks.geojson`, including `"color"`, `"area_name"`, and `"text_found"` for each polygon.

## Usage

1. **Set Credentials**  
   - Provide your **AzureOpenAI** keys and **Vision** endpoint/key in the script:
     ```python
     api_key = "ENTER GPT API KEY"
     endpoint = "ENTER IMAGE ANALYSIS ENDPOINT"
     key = "ENTER IMAGE ANALYSIS CLIENT KEY"
     ```
2. **Install Dependencies**  
   - Required packages include:  
     ```
     pymupdf,
     opencv-python,
     numpy,
     torch,
     rasterio,
     shapely,
     azure-ai-vision,
     ...
     ```
3. **Run the Script**  
   - Adjust `root_dir` to point to your dataset (e.g., `"dataset_with_jgw/aalesund/FOKUS"`)
   - Adjust `output_folder` as desired (e.g., `geojson_outputs/aalesund`).
   - Invoke the script: `python sam2_segmentation_pipeline.py`
4. **Outputs**  
   - For each image, you get:
     - **`<basename>_drawn.jpg`**: The image with drawn polygons.
     - **`<basename>_masks.geojson`**: Mask polygons and properties.

## Notes and Caveats

- If a TIFF file lacks CRS or uses something other than EPSG:25832, it is labeled `"tif_bad"` and assigned no real transform.
- If a JPEG lacks `.jgw`, it is `"jpg_no_jgw"`.
- GPT-4 Vision calls rely on a PDF “tegnforklaring” matching the image’s base name plus `_tegnforklaring.pdf`.
- Azure Vision calls can be **slow** or fail with network issues. Recognized text is stored as `"text_found"`.
- If no transform is found, polygons are written in pixel coordinates using an **identity** transform `(A=1, E=1)`.
