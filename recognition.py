import os
import cv2
from ultralytics import YOLO

# 訓練好的模型
model = YOLO('models/best3.pt')

# 定義類別名稱
class_names = {
    0: 'carroteggs',
    1: 'chickennuggets',
    2: 'sausage',
    3: 'curry',
    4: 'frieddumplings',
    5: 'friedeggs',
    6: 'beansprouts',
    7: 'rice',
    8: 'Potatocake',
    9: 'waterspinach',
    10: 'Friedchickenlegs',
    11: 'brocoli',
    12: 'cabbage',
    13: 'DONTUSE',
    14: 'Friedchickensteak'
}

# 圖片資料夾
image_folder = 'D:\\yolov8-food\\test\\input'
# 結果輸出資料夾
output_folder = 'D:\\yolov8-food\\test\\output'
os.makedirs(output_folder, exist_ok=True)

# 獲取所有圖片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        # 模型推理
        results = model(image)

        # 繪製標註
        annotated_image = results[0].plot()

        print(f"Results for {image_file}:")
        for result in results[0].boxes:
            cls = int(result.cls)
            class_name = class_names.get(cls, "Unknown")
            conf = float(result.conf)
            box = result.xyxy.cpu().numpy().tolist()  # 將張量轉換為列表
            print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {box}")

        # 保存標註圖片
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, annotated_image)

cv2.destroyAllWindows()
