import os
import sys
from paddleocr import PaddleOCR
from PIL import Image
from pdf2image import convert_from_path

# --- CONFIGURATION ---
INPUT_FILE = 'input.pdf'        # <--- Your uploaded PDF file
PROCESSED_FILE = 'ready_for_ocr.jpg'
OUTPUT_FOLDER = './output'
MAX_DIMENSION = 2000
MAX_SIZE_KB = 400

def prepare_input(input_path, temp_image_path):
    """
    Handles both PDF and Image inputs.
    If PDF: Converts 1st page to JPG.
    If Image: Just copies/opens it.
    Then compresses to strict limits.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found.")
        return False

    try:
        # Check if PDF
        if input_path.lower().endswith('.pdf'):
            print(f"📄 PDF detected. Converting first page to image...")
            # Convert first page only (saving RAM)
            pages = convert_from_path(input_path, first_page=1, last_page=1)
            img = pages[0].convert('RGB')
        else:
            # Assume it's an image
            img = Image.open(input_path).convert('RGB')

        # --- COMPRESSION LOGIC ---
        # Resize logic
        width, height = img.size
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            print(f"📉 Resizing from {width}x{height} to under {MAX_DIMENSION}px...")
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        
        # Compression loop
        quality = 95
        while quality > 10:
            img.save(temp_image_path, 'JPEG', quality=quality, dpi=(96, 96))
            file_size_kb = os.path.getsize(temp_image_path) / 1024
            if file_size_kb <= MAX_SIZE_KB:
                print(f"✅ Ready for OCR: {file_size_kb:.1f}KB | Quality: {quality}")
                return True
            quality -= 5 
            
        print("⚠️ Warning: Could not compress under 200KB, but proceeding.")
        return True

    except Exception as e:
        print(f"❌ Error preparing file: {e}")
        return False

# --- MAIN EXECUTION ---

# 1. Create output directory
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Processing {INPUT_FILE}...")

# 2. Convert & Compress
if prepare_input(INPUT_FILE, PROCESSED_FILE):

    # 3. Initialize Heavy V5 Model
    print("\n⏳ Loading Heavy V5 Model...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # 4. Run Inference
    print(f"\n🚀 Running Inference on {PROCESSED_FILE}...")
    try:
        result = ocr.predict(PROCESSED_FILE)

        print("\n💾 Saving results...")
        for res in result:
            # Save JSON and Image to output folder
            res.save_to_json(OUTPUT_FOLDER)
            res.save_to_img(OUTPUT_FOLDER)

        print(f"✅ Done! Check the '{OUTPUT_FOLDER}' folder for your JSON.")
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")

else:
    print("Skipping OCR due to file error.")