import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

TXT_DIRS = [
    r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\Annotations_txt',
    r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\TXT'
]
JPG_DIRS = [
    r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\JPEGImages',
    r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\jpg'
]
OUTPUT_FILE = r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\mask_train.txt'

def find_image(txt_file):
    base = os.path.splitext(txt_file)[0]
    for jpg_dir in JPG_DIRS:
        for ext in ['.jpg', '.png']:
            img_path = os.path.join(jpg_dir, base + ext)
            if os.path.exists(img_path):
                return img_path
    return None

def process_annotation(txt_dir, txt_file):
    txt_path = os.path.join(txt_dir, txt_file)
    img_path = find_image(txt_file)
    
    if not img_path:
        return None
    
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    with open(txt_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        
        class_map = {4: 0, 3: 1, 0: 0, 1: 1}
        class_id = class_map.get(int(class_id), int(class_id))
        
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)
        
        boxes.append(f"{x1},{y1},{x2},{y2},{int(class_id)}")
    
    if boxes:
        return f"{img_path} {' '.join(boxes)}"
    return None

if __name__ == '__main__':
    all_txt_files = []
    for txt_dir in TXT_DIRS:
        txt_files = os.listdir(txt_dir)
        for f in txt_files:
            if f.endswith('.txt'):
                all_txt_files.append((txt_dir, f))
    
    print(f"Found {len(all_txt_files)} txt files total")

    BATCH_SIZE = 100
    num_batches = (len(all_txt_files) + BATCH_SIZE - 1) // BATCH_SIZE

    all_results = []
    seen = set()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_txt_files))
        batch_files = all_txt_files[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)...")
        
        batch_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_annotation, d, f): (d, f) for d, f in batch_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result and result not in seen:
                    seen.add(result)
                    batch_results.append(result)
        
        all_results.extend(batch_results)
        print(f"Batch {batch_idx + 1} completed: {len(batch_results)}/{len(batch_files)} files processed")

    with open(OUTPUT_FILE, 'w') as f:
        for line in all_results:
            f.write(line + '\n')

    print(f"\nAll done! Total: {len(all_results)} records written to {OUTPUT_FILE}")
