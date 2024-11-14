# SCANAI

**SCANAI** is a project focused on segmenting map images by identifying specific regions and converting them into vectorized polygons. The goal is to automate the segmentation process on hand-marked/hand-annotated map images.

## Project Overview
SCANAI explores various segmentation methods to identify the best approach for converting hand-marked map images into structured, vectorized data.

## Methods Explored

### Segment Anything Model (SAM)

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

### Segment Anything Model 2 (SAM2)

- **Parameter Adjustment**: SAM2 outperforms SAM1 when parameters are fine-tuned.
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


### Standard OpenCV Techniques

- **Standard Techniques**: OpenCV was tested for baseline segmentation but yielded less effective results.
- *Additional notes:*
    - Less effective compared to SAM models
    - May not be suitable for complex or detailed map segmentations
    - Maps being handmarked with different markers makes selecting a proper threshold difficult
- **Things Tried:**
    - Thresholding and edge detection
    - Contour detection


## Status and Key Findings

- **Out-of-the-Box Performance**: SAM1 generally delivers good results with minimal setup.
- **Parameter Tuning**: SAM2 can outperform SAM1, demonstrating that segmentation quality can improve with parameter adjustments.
- **Coordinate Prompting**: SAM with coordinate prompting offers the best precision but is limited to one region per prompt.
- **OpenCV Performance**: OpenCV's standard techniques are not as effective for this specific task.

## Considerations for "Good Enough" Quality

A critical discussion point is determining what level of segmentation accuracy is "good enough." While automatic segmentation is promising, achieving ideal results may require fine-tuning with a custom, annotated dataset, which could demand significant resources.

## Next Steps

- **Georeferencing**: Testing segmentation methods with georeferenced maps to assess applicability in real-world spatial contexts.
- **Refinement**: Exploring annotated datasets and fine-tuning to further enhance model performance.
