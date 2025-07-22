from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# ---------- Deskew ----------
def deskew_image_strict(pil_img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 2, maxLineGap=20)

    if lines is None:
        print("[⚠️] No lines found. Skipping deskew.")
        return pil_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        print("[⚠️] No valid angles detected.")
        return pil_img

    median_angle = np.median(angles)
    print(f"[🧭] Strict deskew angle: {median_angle:.2f}°")

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    return Image.fromarray(rotated).convert("RGB")


# ---------- Enhance (Optimized for Header Clarity for Mistral OCR) ----------
def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)

    # Upscale if image is too small
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)

    width, height = gray.size
    header_height = int(height * 0.3)
    header_box = (0, 0, width, header_height)
    header = gray.crop(header_box)

    # Enhance header contrast and sharpness
    header = ImageEnhance.Contrast(header).enhance(2.2)
    header = ImageEnhance.Sharpness(header).enhance(3.0)

    # Binarize the header
    header_np = np.array(header)
    _, header_thresh = cv2.threshold(header_np, 180, 255, cv2.THRESH_BINARY)
    header = Image.fromarray(header_thresh)

    # Merge enhanced header back
    enhanced_gray = gray.copy()
    enhanced_gray.paste(header, (0, 0))

    # Final enhancement on entire image
    enhanced_gray = ImageEnhance.Contrast(enhanced_gray).enhance(1.3)
    enhanced_gray = ImageEnhance.Sharpness(enhanced_gray).enhance(1.2)

    return enhanced_gray.convert("RGB")


# ---------- /align-image ----------
@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)

        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- /enhance-image ----------
@app.post("/enhance-image")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        enhanced = enhance_image(aligned)

        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=enhanced_image.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- /enhance-to-pdf ----------
@app.post("/enhance-to-pdf")
async def enhance_to_pdf(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        enhanced = enhance_image(aligned)

        pdf_buffer = BytesIO()
        enhanced.save(pdf_buffer, format="PDF")
        pdf_buffer.seek(0)

        return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=enhanced_output.pdf"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
