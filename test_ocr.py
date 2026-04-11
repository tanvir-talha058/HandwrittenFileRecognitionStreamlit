import os
from paddleocr import PaddleOCR

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("Initializing PaddleOCR...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    print("PaddleOCR initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    import traceback
    traceback.print_exc()
