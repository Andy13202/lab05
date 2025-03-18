import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 載入 TensorFlow Lite 模型
model_path = "ssd_mobilenet_v1.tflite"  # 確保這個檔案已經下載
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 取得輸入與輸出張量的詳細資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 讀取標籤檔案（COCO dataset）
labels_path = "coco_labels.txt"  # 需提供標籤檔案
with open(labels_path, "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# 讀取圖片
image_path = "test.jpg"  # 請更換為你的圖片
image = cv2.imread(image_path)

# 確保圖片成功讀取
if image is None:
    print("無法讀取圖片，請確認圖片檔案是否存在。")
    exit()

# 調整圖片大小，使其符合模型的輸入需求 (300x300)
input_shape = input_details[0]['shape']
image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

# 將圖片送入 TensorFlow Lite 模型
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 獲取模型輸出結果
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # 物件邊界框
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # 物件類別
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # 物件分數（信心度）

# 解析並標記圖片上的物件
h, w, _ = image.shape
for i in range(len(scores)):
    if scores[i] > 0.5:  # 只顯示高信心度的結果
        y_min, x_min, y_max, x_max = boxes[i]
        x_min, x_max = int(x_min * w), int(x_max * w)
        y_min, y_max = int(y_min * h), int(y_max * h)

        # 繪製邊界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 標示標籤與信心度
        label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 顯示結果
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
