from ultralytics import YOLO
import cv2
import os

model = YOLO('best.pt')

# โฟลเดอร์เก็บรูปผลลัพธ์
os.makedirs('fallback_demo', exist_ok=True)

# ใส่ path รูปไพ่ที่มีอยู่ในเครื่อง อย่างน้อย 3 รูป ต่างไพ่กัน
test_images = [
    'test1.jpg',   # เปลี่ยนเป็นชื่อไฟล์จริง
    'test2.jpg',
    'test3.jpg',
]

for i, img_path in enumerate(test_images):
    if not os.path.exists(img_path):
        print(f"ไม่เจอไฟล์: {img_path} ข้ามไป")
        continue

    results = model(img_path, conf=0.5)

    # วาด bounding boxes
    annotated = results[0].plot()

    # resize เป็น 1280x720 ตามที่ QC checklist กำหนด
    annotated_resized = cv2.resize(annotated, (1280, 720))

    output_path = f'fallback_demo/fallback_{i+1}.jpg'
    cv2.imwrite(output_path, annotated_resized)

    print(f"\nรูปที่ {i+1}: {img_path}")
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {label}: {conf:.1%}")

print("\nบันทึกรูปทั้งหมดใน fallback_demo/ แล้ว")
print("ส่งไฟล์ใน folder นี้ให้เพื่อน #3 สำหรับ Slide 12 ได้เลย")
