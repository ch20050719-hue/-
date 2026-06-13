import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

ANNOTATIONS_DIR = r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\Annotations'
JPEG_IMAGES_DIR = r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\JPEGImages'
OUTPUT_DIR = r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\Annotations_txt'
CLASSES = {'mask': 0, 'nomask': 1}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_xml_to_txt(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        filename = root.find('filename').text
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        
        lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASSES:
                continue
            class_id = CLASSES[name]
            
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if lines:
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(txt_filename))
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            return True
        return False
    except Exception as e:
        print(f"Error processing {xml_file}: {str(e)}")
        return False

xml_files = [os.path.join(ANNOTATIONS_DIR, f) for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
print(f"Found {len(xml_files)} XML files")

BATCH_SIZE = 100
num_batches = (len(xml_files) + BATCH_SIZE - 1) // BATCH_SIZE

total_success = 0
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(xml_files))
    batch_files = xml_files[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)...")
    
    batch_success = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_xml_to_txt, xml_file): xml_file for xml_file in batch_files}
        
        for future in as_completed(futures):
            if future.result():
                batch_success += 1
    
    total_success += batch_success
    print(f"Batch {batch_idx + 1} completed: {batch_success}/{len(batch_files)} successful")

print(f"\nAll done! Total: {total_success}/{len(xml_files)} files converted successfully")
print(f"Output directory: {OUTPUT_DIR}")
