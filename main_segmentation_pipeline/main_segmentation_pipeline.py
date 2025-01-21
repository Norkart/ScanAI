import os
import io
import base64
import cv2
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon, mapping
import geojson
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import fitz
from PIL import Image, ImageEnhance
import webcolors
from openai import AzureOpenAI
import shapely.geometry
import concurrent.futures
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis.models import VisualFeatures
import re
from concurrent.futures import ThreadPoolExecutor
import sys
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Azure GPT-4 Vision
# Please fill in these with your actual credentials:
api_key = "ENTER GPT API KEY"
api_version = "ENTER GPT API VERSION"
api_base = "ENTER GPT API BASE"  # Base URL like "https://your-resource.openai.azure.com"
deployment_name = "gpt-4"  # e.g. "gpt-4"


# Initialize the AzureOpenAI client
openAIClient = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

endpoint = "ENTER IMAGE ANALYSIS ENDPOINT"
key = "ENTER IMAGE ANALYSIS CLIENT KEY"

vision_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)
# ------------------------------------------------------------------------------
# 1) PDF -> base64 for GPT-4 Vision
# ------------------------------------------------------------------------------
def pdf_to_jpg_base64(pdf_path, output_file="enhanced_image.jpg", contrast_factor=1.5):
    """
    Convert the first page of a PDF to a JPG, enhance its contrast, save it to a file,
    and return the base64-encoded string. Returns None on failure.
    """
    try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)

        # Render the first page
        page = pdf_document[0]  # First page (index 0)
        pixmap = page.get_pixmap()

        # Convert pixmap to an image using PIL
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        # Save the enhanced image to a BytesIO buffer
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        # Encode the image as base64
        base64_image = base64.b64encode(buf.read()).decode("utf-8")
        return base64_image

    except Exception as e:
        return None

# ------------------------------------------------------------------------------
# 2) GPT-4 call with color + tegnforklaring
# ------------------------------------------------------------------------------
def gpt4_get_area_name_from_hex(pdf_path, hex_color):
    """
    1) pdf_to_jpg_base64(pdf_path)
    2) Prompt GPT-4 Vision with color + PDF
    3) Return a single area name
    """
    pdf_b64 = pdf_to_jpg_base64(pdf_path)
    if not pdf_b64:
        return "Unknown - No legend provided"

    prompt_text = (
        f"You are given a legend in the attached image, and the user has provided the color {hex_color}. "
        "Analyze the legend carefully to identify the area name that most closely matches this color. "
        "The area names will always appear to the left or right of their corresponding color in the legend, not above or below. "
        "Return ONLY the single most accurate matching area name, in lowercase, without any additional explanation, formatting, or punctuation.\n\n"
        "Additional context if you are uncertain:\n"
        "- Shades of yellow/tan/beige are often associated with 'eneboliger', 'eneboligbebyggelse', or 'villamessig bebyggelse'.\n"
        "- Shades of orange often indicate 'konsentrert småhusbebyggelse' or 'konsentrert boligbebyggelse rekkehus'.\n"
        "- Shades of dark orange/reddish frequently map to 'blokkbebyggelse' or 'boligbebyggelse høyhus'.\n"
        "- Shades of green or paleish greenish yellow typically correspond to 'friområdet', 'landbruk', 'jordbruksomådet', or 'fareområdet'.\n"
        "- Shades of blue or blueish green are usually 'forretningsbebyggelse' or 'water'.\n"
        "- Shades of pink or pinkish red often indicate 'off. bebyggekse institusjoner', 'offentlig bebyggelse', 'offentlig areal', or 'spesialområdet'.\n"
        "- Shades of purple are usually 'industri' or 'industriområdet'.\n\n"
        "Lets think things through and take your time and really analyze it. Analyze both the hex color code and the color name as well as the additional context."
    )

    try:
        # Real call to Azure GPT-4 Vision
        response = openAIClient.chat.completions.create(
            model=deployment_name,  # or "gpt-4-vision" if your model is named differently
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{pdf_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=128
        )
        # Extract the reply
        area_name = response.choices[0].message.content.strip()
        return area_name
    except Exception as e:
        print("GPT ERROR", e)
        return "Unknown - GPT api error"

# ------------------------------------------------------------------------------
# 3) JGW + TIF loading
# ------------------------------------------------------------------------------
def read_jgw(jgw_file):
    """
    Return (A,B,C,D,E,F) from a standard JGW file or None on fail.
    """
    try:
        with open(jgw_file,'r') as f:
            lines=f.read().splitlines()
        A=float(lines[0])
        D=float(lines[1])
        B=float(lines[2])
        E=float(lines[3])
        C=float(lines[4])
        F=float(lines[5])
        return (A,B,C,D,E,F)
    except:
        return None

def pixel_to_world(px, py, A, B, C, D, E, F):
    X = A*px + B*py + C
    Y = D*px + E*py + F
    return (X, Y)


# We want to define how to crop each mask region for analysis
def crop_mask_region(image_bgr, mask_2d):
    """
    1) Identify the bounding box of 'mask_2d'.
    2) Create a sub-image, blank out everything outside the mask (white).
    3) Return sub-image as in-memory JPEG bytes or None if out of valid dimension range.
    """

    coords = np.where(mask_2d > 0)
    if coords[0].size == 0:
        return None

    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()

    sub_img = image_bgr[min_y:max_y+1, min_x:max_x+1].copy()

    # Dimension check
    # Skip if outside Azure Vision’s valid range: 50px <= dimension <= 16000px
    h, w = sub_img.shape[:2]
    if h < 50 or w < 50 or h > 16000 or w > 16000:
        return None

    local_mask = np.zeros(sub_img.shape[:2], dtype=np.uint8)
    shifted_y = coords[0] - min_y
    shifted_x = coords[1] - min_x
    local_mask[shifted_y, shifted_x] = 255

    # blank out outside
    sub_img[local_mask == 0] = (255, 255, 255)

    # Encode to bytes
    try:
        is_success, buffer = cv2.imencode(".jpg", sub_img)
        if not is_success:
            return None
        return buffer.tobytes()
    except:
        return None


def analyze_mask_polygon(polygon_dict, image_bgr):
    """
    1) Crop sub-image from mask.
    2) Skip if too small/large or if we fail to encode.
    3) Call Azure Vision with 'READ' to get text.
    """
    seg_mask = polygon_dict['mask']
    image_data = crop_mask_region(image_bgr, seg_mask)
    if image_data is None:
        # dimension out of range or no sub-image
        return ""

    visual_features = [
        VisualFeatures.READ
    ]

    try:
        result = vision_client.analyze(
            image_data=image_data,
            visual_features=visual_features,
            # you can omit 'gender_neutral_caption' if only using READ
            language="en"
        )
    except Exception as e:
        print(f"Azure Vision analyze error: {e}")
        return ""

    # If region doesn't produce read_result, skip
    found_texts = []
    if result.read and result.read.blocks:
        # Loop through each block
        for block in result.read.blocks:
            # Then loop through each line in that block
            if block.lines:
                for line in block.lines:
                    # Append just the textual content
                    found_texts.append(line.text)
    return "\n".join(found_texts)


# ------------------------------------------------------------------------------
# 5) The main pipeline
# ------------------------------------------------------------------------------
def segment_and_generate_geojson_unified(
    original_image_bgr,
    transform_params,
    input_basename,
    mask_generator,
    output_folder,
    original_path,
    color_snap_threshold=5.0
):
    """
    1) Segment the image with SAM2.
    2) Filter & find polygons.
    3) In parallel, find the "dominant color cluster" for each polygon.
       - Clusters merge pixels whose BGR differs by <~4 in each channel,
         effectively ignoring tiny color variations (1–3 off).
       - Skip black/gray if a colorful cluster is available.
    4) Collect unique colors -> call GPT-4 in parallel for each color.
    5) Save color & area_name in GeoJSON.
    """

    # Keep a copy for drawing
    image_for_drawing = original_image_bgr.copy()

    # Step A: Segment
    image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)
    print(f"{input_basename}: total raw masks = {len(sam_result)}")

    def is_mask_inside(outer_mask, inner_mask):
        return np.all(outer_mask[inner_mask > 0])

    def is_colorful_region(image, mask, sat_thr=20, bri_thr=50):
        """
        Checks if a mask region is 'colorful enough' in HSV, to filter out
        near-black or near-white.
        """
        if mask.ndim == 2:
            mask = mask.astype(np.uint8)
        else:
            raise ValueError("mask must be 2D")

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_255 = mask * 255
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masked = cv2.bitwise_and(hsv, hsv, mask=mask_255)

        s_chan = masked[..., 1]
        v_chan = masked[..., 2]

        mean_s = cv2.mean(s_chan, mask=mask_255)[0]
        mean_v = cv2.mean(v_chan, mask=mask_255)[0]

        return (mean_s > sat_thr) and (mean_v > bri_thr)

    # B) Filter
    masks_with_areas = []
    for i, seg_info in enumerate(sam_result):
        seg = seg_info['segmentation']
        if np.any(seg):
            area = np.sum(seg)
            coords = np.argwhere(seg)
            if coords.shape[0] == 0:
                continue
            min_y, max_y = coords[:,0].min(), coords[:,0].max()
            min_x, max_x = coords[:,1].min(), coords[:,1].max()
            masks_with_areas.append((i, seg, area, (min_x, min_y, max_x, max_y)))

    # Sort largest -> smallest
    masks_with_areas.sort(key=lambda x: x[2], reverse=True)
    c_thresh = int(0.1 * len(masks_with_areas))
    to_remove = set()

    for idx, (oi, om, oa, ob) in enumerate(masks_with_areas):
        oxmin, oymin, oxmax, oymax = ob
        contained_count = 0
        for (ii, im, ia, ib) in masks_with_areas[idx+1:]:
            ixmin, iymin, ixmax, iymax = ib
            if (ixmin >= oxmin and ixmax <= oxmax and iymin >= oymin and iymax <= oymax):
                if is_mask_inside(om, im):
                    contained_count += 1
        if contained_count >= c_thresh and c_thresh != 0:
            to_remove.add(oi)

    # Keep only valid
    filtered_masks = [(i,m,a,b) for (i,m,a,b) in masks_with_areas if i not in to_remove]
    h, w = original_image_bgr.shape[:2]
    image_area = h*w
    filtered_masks = [(i,m,a,b) for (i,m,a,b) in filtered_masks if a<image_area]
    filtered_masks = [
        (i,m,a,b)
        for (i,m,a,b) in filtered_masks
        if is_colorful_region(original_image_bgr, m, 0, 0)
    ]
    final_sam = [sam_result[i] for (i,_,_,_) in filtered_masks]
    print(f"{input_basename}: total masks after filtering = {len(final_sam)}")

    def smooth_contour(cont, ws=15):
        if ws%2==0:
            ws+=1
        half=ws//2
        cont=np.concatenate((cont[-half:], cont, cont[:half]), axis=0)
        out=[]
        for x in range(half, len(cont)-half):
            block=cont[x-half : x+half+1]
            mp=np.mean(block,axis=0)
            out.append(mp)
        return np.array(out,dtype=np.int32)

    polygons=[]
    for md in final_sam:
        seg=md['segmentation'].astype(np.uint8)
        conts,_=cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not conts:
            continue
        for c in conts:
            if c.shape[0]<5:
                continue
            c=c.reshape(-1,2)
            sc=smooth_contour(c,15)
            if sc.shape[0]>=3:
                shp=Polygon(sc)
                if not shp.is_valid or shp.area==0:
                    shp=shp.buffer(0)
                    if not shp.is_valid or shp.area==0:
                        continue
                polygons.append({'polygon':shp,'mask':seg,'area':shp.area})

    polygons = [p for p in polygons if p['area']<0.1*image_area]
    polygons.sort(key=lambda x:x['area'], reverse=True)

    final_list=[]
    def mostly_inside(poly, existing, thr=0.95):
        for ep in existing:
            inter_area=poly.intersection(ep['polygon']).area
            if poly.area==0:
                continue
            if (inter_area/poly.area)>=thr:
                return True
        return False

    for p in polygons:
        if not mostly_inside(p['polygon'], final_list):
            final_list.append(p)
    print(f"{input_basename}: polygons after final overlap filtering = {len(final_list)}")

    # Step: draw
    image_draw = image_for_drawing.copy()
    for p in final_list:
        pol = p['polygon']
        if isinstance(pol, Polygon):
            coords = np.array(list(pol.exterior.coords), dtype=np.int32)
            cv2.polylines(image_draw, [coords], True, (0,255,0), 3)
        else:
            for sp in pol.geoms:
                coords = np.array(list(sp.exterior.coords), dtype=np.int32)
                cv2.polylines(image_draw, [coords], True, (0,255,0), 3)
    out_drawn=os.path.join(output_folder, f"{input_basename}_drawn.jpg")
    cv2.imwrite(out_drawn,image_draw)
    print(f"Saved mask outlines to {out_drawn}")

    if transform_params:
        A,B,C,D,E,F = transform_params
    else:
        A = 1.0
        B = 0.0
        C = 0.0
        D = 0.0
        E = 1.0
        F = 0.0

    def cluster_pixel(b, step=4):
        """
        Round b to nearest multiple of 'step'.
        E.g. if step=4, 0..3 -> 0, 4..7 -> 4, etc.
        """
        # We can do integer rounding or floor. We'll do floor.
        return (b//step)*step

    def is_grayish(bgr_int, threshold=30):
        b = (bgr_int>>16)&0xFF
        g = (bgr_int>>8)&0xFF
        r = bgr_int&0xFF
        return (max(b,g,r)-min(b,g,r))<threshold

    def compute_dominant_colorful_pixel(seg_mask):
        """
        1) gather all pixels in seg_mask
        2) cluster them in increments of 4 to ignore tiny color differences
        3) pick largest cluster
        4) if that cluster is gray, see if there's a large cluster that isn't
        5) fallback to top if no non-gray cluster exists
        """
        seg_mask=seg_mask.astype(np.uint8)
        coords=np.where(seg_mask>0)
        if coords[0].size==0:
            return "#000000"
        region_pixels=original_image_bgr[coords[0],coords[1]].reshape(-1,3).astype(np.uint16)

        # cluster them
        cluster_counts={}
        for (b,g,r) in region_pixels:
            b_clust = cluster_pixel(b)
            g_clust = cluster_pixel(g)
            r_clust = cluster_pixel(r)
            comb = (b_clust<<16) + (g_clust<<8) + r_clust
            cluster_counts[comb] = cluster_counts.get(comb, 0) + 1

        # sort by freq desc
        cluster_items = sorted(cluster_counts.items(), key=lambda x:x[1], reverse=True)

        # if top is non-gray => done. else see if there's a big non-gray
        top_key,top_count = cluster_items[0]
        total_px = coords[0].size

        # check if there's a non-gray cluster
        # if top is gray => see if there's any non-gray that is big enough to overshadow
        # if ANY non-gray cluster exists, pick the largest among them. else pick top
        # define a separate list for non-gray
        non_gray_clusters=[(k,v) for (k,v) in cluster_items if not is_grayish(k)]
        if len(non_gray_clusters)>0:
            # pick largest non-gray
            c_key, c_count = non_gray_clusters[0]  # largest in freq
            b=(c_key>>16)&0xFF
            g=(c_key>>8)&0xFF
            r=c_key&0xFF
            return f"#{r:02X}{g:02X}{b:02X}"
        else:
            # everything is gray => pick top cluster
            b=(top_key>>16)&0xFF
            g=(top_key>>8)&0xFF
            r=top_key&0xFF
            return f"#{r:02X}{g:02X}{b:02X}"

    def compute_color_task(p_item):
        return compute_dominant_colorful_pixel(p_item['mask'])

    # parallel
    with ThreadPoolExecutor() as executor:
        color_futures = [executor.submit(compute_color_task,p) for p in final_list]
        polygon_colors = [f.result() for f in color_futures]

    for i, c_hex in enumerate(polygon_colors):
        final_list[i]['color_hex']=c_hex

    # PART 2: GPT calls for unique colors
    unique_colors=set(p['color_hex'] for p in final_list)
    pdf_tegn=os.path.splitext(os.path.basename(original_path))[0]+"_tegnforklaring.pdf"
    pdf_path=os.path.join(os.path.dirname(original_path),pdf_tegn)

    color_to_area={}

    def gpt_area_task(c_hex):
        return gpt4_get_area_name_from_hex(pdf_path, c_hex)

    with ThreadPoolExecutor() as executor:
        gpt_futs={
            c_hex:executor.submit(gpt_area_task,c_hex)
            for c_hex in unique_colors
        }
        for c_hex,fut in gpt_futs.items():
            color_to_area[c_hex]=fut.result()

    with ThreadPoolExecutor(max_workers=5) as executor:
        # submit for each polygon
        vision_futs = [
            executor.submit(analyze_mask_polygon, p, original_image_bgr)
            for p in final_list
        ]
        # gather results
        text_results = [f.result() for f in vision_futs]
    
    # store text in polygon dict
    for i, txt in enumerate(text_results):
        final_list[i]['text_found'] = txt

    # Convert polygons -> world coords
    def polygon_to_world_coords(geom, A, B, C, D, E, F):
        if isinstance(geom, Polygon):
            exterior_coords = []
            for px, py in geom.exterior.coords:
                xw, yw = pixel_to_world(px, py, A, B, C, D, E, F)
                exterior_coords.append((xw, yw))
    
            interior_list = []
            for interior in geom.interiors:
                hole = []
                for px2, py2 in interior.coords:
                    x2, y2 = pixel_to_world(px2, py2, A, B, C, D, E, F)
                    hole.append((x2, y2))
                interior_list.append(hole)
    
            return Polygon(exterior_coords, interior_list)
    
        elif isinstance(geom, MultiPolygon):
            new_subpolys = []
            for subpoly in geom.geoms:
                # Convert each sub-polygon's exterior
                exterior_coords = []
                for px, py in subpoly.exterior.coords:
                    xw, yw = pixel_to_world(px, py, A, B, C, D, E, F)
                    exterior_coords.append((xw, yw))
    
                # Convert interiors
                interior_list = []
                for interior in subpoly.interiors:
                    hole = []
                    for px2, py2 in interior.coords:
                        x2, y2 = pixel_to_world(px2, py2, A, B, C, D, E, F)
                        hole.append((x2, y2))
                    interior_list.append(hole)
    
                poly_converted = Polygon(exterior_coords, interior_list)
                # Keep only valid, non-empty polygons
                if poly_converted.is_valid and not poly_converted.is_empty:
                    new_subpolys.append(poly_converted)
    
            return MultiPolygon(new_subpolys) if new_subpolys else MultiPolygon()
    
        else:
            return geom

    features=[]
    for p in final_list:
        poly=p['polygon']
        c_hex=p['color_hex']
        area_name=color_to_area[c_hex]
        found_text = p.get('text_found', '')
        w_poly=polygon_to_world_coords(poly,A,B,C,D,E,F)
        if w_poly.is_valid and not w_poly.is_empty:
            ft=geojson.Feature(
                geometry=mapping(w_poly),
                properties={
                    "color":c_hex,
                    "area_name":area_name,
                    "text_found":found_text
                }
            )
            features.append(ft)

    fc=geojson.FeatureCollection(features)
    out_geojson=os.path.join(output_folder,f"{input_basename}_masks.geojson")
    with open(out_geojson,'w',encoding='utf-8') as f:
        geojson.dump(fc,f,indent=2,ensure_ascii=False)

    print(f"Saved {len(features)} polygons to {out_geojson}.")
    print("Done with", input_basename)


# ------------------------------------------------------------------------------
# 7) The main driver
# ------------------------------------------------------------------------------
class ImageItem:
    """
    Holds:
      - path (str)
      - image_type (str like 'jpg_sos', 'jpg_no_sos', 'tif_ok', 'tif_bad', 'pdf', 'png', etc.)
      - image_bgr (np.ndarray or None)
      - transform_params (tuple or None) => (A,B,C,D,E,F)
    """
    def __init__(self, path: str, image_type: str, image_bgr, transform_params):
        self.path = path
        self.image_type = image_type
        self.image_bgr = image_bgr
        self.transform_params = transform_params
    
    def __repr__(self):
        tform_str = "YES" if self.transform_params else "NO"
        return f"<ImageItem path='{self.path}', type='{self.image_type}', hasTransform={tform_str}>"
    
def load_image_jpg(full_path):
    """
    Attempt to load .jpg as image_bgr => np.ndarray (BGR).
    If .jgw found => parse (A,B,C,D,E,F).
    Return (image_bgr, transform_params) or (None, None).
    """
    img = cv2.imread(full_path)
    if img is None:
        return None,None
    
    jgw_path = os.path.splitext(full_path)[0]+".jgw"
    if os.path.exists(jgw_path):
        coords = read_jgw(jgw_path)
        return img, coords
    else:
        # no transform
        return img, None

def load_image_tif(full_path):
    """
    Return (image_bgr, transform) if EPSG:25832, else (image_bgr, None).
    If we cannot open => (None, None).
    """
    try:
        with rasterio.open(full_path) as ds:
            bc=min(ds.count,3)
            data=ds.read()
            arr=np.transpose(data[:bc,:,:],(1,2,0)).astype(np.uint8)
            if bc==1:
                arr=cv2.merge([arr[:,:,0],arr[:,:,0],arr[:,:,0]])
            elif bc==2:
                arr=cv2.merge([arr[:,:,0],arr[:,:,1],arr[:,:,1]])
            else:
                arr=cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            # Check CRS
            if ds.crs and str(ds.crs)=="EPSG:25832":
                A=ds.transform.a
                B=ds.transform.b
                C=ds.transform.xoff
                D=ds.transform.d
                E=ds.transform.e
                F=ds.transform.yoff
                coords=(A,B,C,D,E,F)
                return arr, coords
            else:
                # load image but no transform
                return arr, None
    except:
        return None,None

def load_image_pdf(full_path):
    """
    Return (image_bgr, None) for first page of PDF or (None,None) on fail.
    """
    try:
        doc=fitz.open(full_path)
        page=doc[0]
        pix=page.get_pixmap()
        np_img = np.array(Image.frombytes("RGB",(pix.width,pix.height),pix.samples))[:,:,::-1]
        return np_img,None
    except:
        return None,None

def load_image_png(full_path):
    """
    Return (image_bgr, None).
    """
    img=cv2.imread(full_path)
    return img,None

def find_all_supported_images(root_dir):
    """
    Search 'root_dir' recursively for images:
      .jpg / .jpeg
      .tif
      .pdf
      .png
    Return list of ImageItem, each with (path, type, image_bgr, transform_params).
    e.g. type: 'jpg_sos', 'jpg_no_sos', 'tif_ok', 'tif_bad', 'pdf', 'png'
    If we cannot load => skip.
    """

    valid_extensions = ('.jpg', '.jpeg', '.tif', '.pdf', '.png')
    exclude_pattern = re.compile(r"planbestemmelser|tegnforklaring|bestemmelser", re.IGNORECASE)


    results = []

    for sb, dirs, files in os.walk(root_dir):
        files.sort()
        for fname in files:
            lower_ext = os.path.splitext(fname)[1].lower()
            if lower_ext not in valid_extensions:
                continue
            if exclude_pattern.search(fname):
                continue
            
            full_path=os.path.join(sb,fname)
            base,ext=os.path.splitext(fname)
            ext=ext.lower()

            if ext in('.jpg','.jpeg'):
                # attempt load
                img,coords=load_image_jpg(full_path)
                if img is None:
                    continue
                if coords is not None:
                    # so it's 'jpg_sos'
                    results.append(ImageItem(
                        path=full_path,
                        image_type='jpg_jgw',
                        image_bgr=img,
                        transform_params=coords
                    ))
                else:
                    results.append(ImageItem(
                        path=full_path,
                        image_type='jpg_no_jgw',
                        image_bgr=img,
                        transform_params=None
                    ))

            elif ext=='.tif':
                img,coords=load_image_tif(full_path)
                if img is None:
                    # skip
                    continue
                if coords is not None:
                    # correct crs
                    results.append(ImageItem(
                        path=full_path,
                        image_type='tif_ok',
                        image_bgr=img,
                        transform_params=coords
                    ))
                else:
                    results.append(ImageItem(
                        path=full_path,
                        image_type='tif_bad',
                        image_bgr=img,
                        transform_params=None
                    ))

            elif ext=='.pdf':
                img,_=load_image_pdf(full_path)
                if img is None:
                    continue
                results.append(ImageItem(
                    path=full_path,
                    image_type='pdf',
                    image_bgr=img,
                    transform_params=None
                ))

            elif ext=='.png':
                img,_=load_image_png(full_path)
                if img is None:
                    continue
                results.append(ImageItem(
                    path=full_path,
                    image_type='png',
                    image_bgr=img,
                    transform_params=None
                ))
            # else skip

    # Sort by path
    results.sort(key=lambda x: x.path.lower())
    return results

if __name__=="__main__":

    # bfloat16 for entire script
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major>=8:
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build mask generator
    mask_generator=SAM2AutomaticMaskGenerator.from_pretrained(
        model_id="facebook/sam2-hiera-large",
        points_per_side=32,
        points_per_batch=4,
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
    print("Model init done.")

    # Provide your dataset root
    
    root_dir="dataset_with_jgw/aalesund"
    output_folder="main_segmentation_outputs/aalesund"
    os.makedirs(output_folder, exist_ok=True)

    all_imgs = find_all_supported_images(root_dir)

    print("All supported images found.")

    processed_count = {
        "jpg_jgw": 0,
        "jpg_no_jgw": 0,
        "tif_ok": 0,
        "tif_bad": 0,
        "pdf": 0,
        "png": 0
    }

    for item in all_imgs:
        if ".ipynb_checkpoint" in item.path.lower():
            continue

        base_name = os.path.splitext(os.path.basename(item.path))[0]
        print(f"\n[Processing] {item.path}")
    
        # Pull the loaded image data and transform (if any) from the ImageItem
        image_bgr = item.image_bgr
        transform_params = item.transform_params
    
        # If image is None, we skip
        if image_bgr is None:
            continue
    
        segment_and_generate_geojson_unified(
            original_image_bgr=image_bgr,
            transform_params=transform_params,
            input_basename=base_name,
            mask_generator=mask_generator,
            output_folder=output_folder,
            original_path=item.path,
            color_snap_threshold=5.0
        )

        processed_count[item.image_type] += 1
    

    
    summary_text = (
        "=== Processing Summary ===\n"
        f"Number of JPG with coordinates (JGW) processed: {processed_count['jpg_jgw']}\n"
        f"Number of normal JPG (no coords) processed:     {processed_count['jpg_no_jgw']}\n"
        f"Number of valid TIF (EPSG:25832) processed:     {processed_count['tif_ok']}\n"
        f"Number of invalid TIF (no coords) processed:    {processed_count['tif_bad']}\n"
        f"Number of PDF processed:                        {processed_count['pdf']}\n"
        f"Number of PNG processed:                        {processed_count['png']}\n"
        "All done!\n"
    )
    
    # Print to console
    print("\n", summary_text)
    
    # Also write to file
    output_summary_path = os.path.join(output_folder, "processing_summary.txt")
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"Summary written to {output_summary_path}")
