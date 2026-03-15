"""
generate_fallback.py
──────────────────────────────────────────────────────────────
สร้าง Demo Screenshots สำหรับ Slide Presentation (AIE223)

วิธีใช้:
  1. แก้ test_images ด้านล่างให้ตรงกับไฟล์รูปจริงในเครื่อง
  2. รัน: python generate_fallback.py
  3. รูปผลลัพธ์จะอยู่ใน fallback_demo/  พร้อมส่งให้เพื่อนใส่ slide ได้เลย

ผลลัพธ์:
  fallback_demo/fallback_1.jpg  (1280×720px, มี bounding boxes)
  fallback_demo/fallback_2.jpg
  fallback_demo/fallback_3.jpg
"""

from ultralytics import YOLO
import cv2
import os

# ─────────────────────────────────────────
# โหลด model
# ─────────────────────────────────────────
model = YOLO("best.pt")

# ─────────────────────────────────────────
# โฟลเดอร์เก็บรูปผลลัพธ์
# ─────────────────────────────────────────
os.makedirs("fallback_demo", exist_ok=True)

# ─────────────────────────────────────────
# ใส่ path รูปไพ่ที่มีอยู่ในเครื่อง
# เปลี่ยนเป็นชื่อไฟล์จริง อย่างน้อย 3 รูป
# ─────────────────────────────────────────
test_images = [
    "test1.jpg",   # ← เปลี่ยนเป็นชื่อไฟล์จริง เช่น "hand1.jpg"
    "test2.jpg",   # ← เปลี่ยนเป็นชื่อไฟล์จริง
    "test3.jpg",   # ← เปลี่ยนเป็นชื่อไฟล์จริง
]

CONFIDENCE_THRESHOLD = 0.5
OUTPUT_SIZE = (1280, 720)   # ขนาด QC checklist กำหนด

# ─────────────────────────────────────────
# วน process ทุกรูป
# ─────────────────────────────────────────
saved = 0

for i, img_path in enumerate(test_images):
    if not os.path.exists(img_path):
        print(f"⚠️  ไม่เจอไฟล์: {img_path}  → ข้ามไป")
        continue

    print(f"\n{'─'*50}")
    print(f"📷  รูปที่ {i+1}: {img_path}")

    results = model(img_path, conf=CONFIDENCE_THRESHOLD)

    # ── Deduplication (เหมือน app.py) ─────
    best_per_class = {}
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        if label not in best_per_class or conf > best_per_class[label]:
            best_per_class[label] = conf

    # ── แสดงผลใน terminal ─────────────────
    if best_per_class:
        for label, conf in sorted(best_per_class.items()):
            print(f"   🃏  {label}: {conf:.1%}")
    else:
        print("   (ไม่พบไพ่ในรูปนี้)")

    # ── วาด bounding boxes + resize → บันทึก ──
    annotated        = results[0].plot()    # BGR image พร้อม boxes
    annotated_resized = cv2.resize(annotated, OUTPUT_SIZE)

    output_path = f"fallback_demo/fallback_{i+1}.jpg"
    cv2.imwrite(output_path, annotated_resized)
    print(f"   ✅  บันทึกแล้ว → {output_path}  ({OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}px)")
    saved += 1

# ─────────────────────────────────────────
# สรุป
# ─────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"✅  บันทึกรูปทั้งหมด {saved} รูป ใน fallback_demo/")
print("📤  ส่งไฟล์ใน fallback_demo/ ให้เพื่อน #3 สำหรับ Slide 12 ได้เลย")
print(f"{'═'*50}")
