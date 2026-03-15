import os
import uuid
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER = "uploads"
# สร้าง folder อัตโนมัติถ้ายังไม่มี — exist_ok=True ป้องกัน error ถ้ามีอยู่แล้ว
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5

model = YOLO("best.pt")

# Routes
@app.route("/", methods=["GET"])
def index():
    """Serve หน้า frontend หลัก"""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    รับไฟล์รูปภาพ → ทำ YOLO inference → คืน JSON

    Request:
        Content-Type : multipart/form-data
        Field name   : "image"

    Response สำเร็จ (HTTP 200):
    {
        "success": true,
        "count": 2,
        "detections": [
            {
                "class": "AC",
                "confidence": 95.2,
                "box": { "left": 120.5, "top": 80.3, "width": 130.2, "height": 270.1 }
            }
        ]
    }

    Response error (HTTP 4xx/5xx):
    { "error": "error message" }
    """

    # ── 1. Validate request ───────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # ── 2. บันทึก temp file ───────────────
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)

    # ── 3. Inference + cleanup ─────────────
    # เหตุผลที่ต้อง try/finally:
    #   - ถ้า inference สำเร็จ  → ลบ temp file หลัง return
    #   - ถ้า inference throw exception → ลบ temp file ก่อน propagate error
    #   - ป้องกัน temp files สะสมใน uploads/ จนเต็ม disk
    try:
        results = model(temp_path, conf=CONFIDENCE_THRESHOLD)

        detections = []
        for result in results:
            for box in result.boxes:
                # ── class name ────────────────────────
                class_name = model.names[int(box.cls[0])]

                # ── confidence: 0-1 → 0-100, round 1 dp ──
                # เหตุผล: Frontend ตกลง format เป็น เปอร์เซ็นต์
                # เช่น 0.9523 → 95.2 (อ่านง่ายกว่าและแสดงผลบนหน้าเว็บได้เลย)
                confidence = round(float(box.conf[0]) * 100, 1)

                # ── bbox: xyxy → {left, top, width, height} ──
                # YOLO คืน xyxy = [x1, y1, x2, y2]  (corner coordinates)
                # Frontend ต้องการ left/top/width/height แบบ CSS box model
                # left   = x1          (ขอบซ้าย)
                # top    = y1          (ขอบบน)
                # width  = x2 - x1     (ความกว้าง)
                # height = y2 - y1     (ความสูง)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bounding_box = {
                    "left":   round(x1, 1),
                    "top":    round(y1, 1),
                    "width":  round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                }

                detections.append(
                    {
                        "class":      class_name,
                        "confidence": confidence,
                        "box":        bounding_box,
                    }
                )

        return jsonify(
            {
                "success":    True,
                "count":      len(detections),
                "detections": detections,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # ลบ temp file เสมอ ไม่ว่าจะสำเร็จหรือ exception
        if os.path.exists(temp_path):
            os.remove(temp_path)



# Entry point
if __name__ == "__main__":
    # debug=True: auto-reload เมื่อแก้โค้ด + แสดง error detail บน browser
    # host='0.0.0.0': รับ connection จากทุก IP (ไม่ใช่แค่ localhost)
    app.run(host="0.0.0.0", port=5000, debug=True)

