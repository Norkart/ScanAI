import os
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.drawing.image import Image

clustered_images_folder = 'clustered_images'

temp_image_folder = 'temp_images'

if not os.path.exists(temp_image_folder):
    os.makedirs(temp_image_folder)

def compress_image(input_path, output_path, size=(100, 100)):
    img = PILImage.open(input_path)
    img = img.convert('RGB')  
    img.thumbnail(size)
    img.save(output_path, format='JPEG', quality=85)

clusters_data = {}

for cluster_name in os.listdir(clustered_images_folder):
    cluster_path = os.path.join(clustered_images_folder, cluster_name)
    if os.path.isdir(cluster_path):
        image_names = [f for f in os.listdir(cluster_path) if os.path.isfile(os.path.join(cluster_path, f))]
        compressed_paths = []
        for img in image_names:
            original_path = os.path.join(cluster_path, img)
            compressed_path = os.path.join(temp_image_folder, f"{cluster_name}_{img}")
            compress_image(original_path, compressed_path)
            compressed_paths.append(compressed_path)
        clusters_data[cluster_name] = {
            "names": image_names,
            "paths": compressed_paths,
        }

def create_excel_with_images(clusters, file_name):
    wb = Workbook()
    ws = wb.active
    ws.title = "Clusters"

    headers = []
    for cluster_name in clusters.keys():
        headers.extend([f"{cluster_name}", f"{cluster_name} Images"])
    ws.append(headers)

    max_rows = max(len(data["names"]) for data in clusters.values())

    for row_idx in range(max_rows):
        row = []
        for cluster_name, data in clusters.items():
            if row_idx < len(data["names"]):
                row.append(data["names"][row_idx])
            else:
                row.append(None)
            row.append(None)
        ws.append(row)

    for col_idx, (cluster_name, data) in enumerate(clusters.items(), start=1):
        for row_idx, image_path in enumerate(data["paths"], start=2):  
            img = Image(image_path)
            img.width = 50  
            img.height = 50  
            ws.add_image(img, ws.cell(row=row_idx, column=col_idx * 2).coordinate)

    wb.save(file_name)
    print(f"Saved {file_name}")

output_excel_file = 'clusters_with_images.xlsx'
create_excel_with_images(clusters_data, output_excel_file)

import shutil
shutil.rmtree(temp_image_folder)
