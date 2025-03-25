import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

ALLOWED_CLASSES = [ 'person', 'oven' ]

# 如果學生沒新增任何類別，就不執行偵測
if not ALLOWED_CLASSES:
    exit()

# 載入 TensorFlow Lite 模型
model_path = "ssd_mobilenet_v1.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 取得輸入與輸出張量的詳細資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 讀取標籤檔案（COCO dataset）
labels_path = "coco_labels.txt"
with open(labels_path, "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# 設定信心度閾值
CONFIDENCE_THRESHOLD = 0.4

# 讀取圖片
image_path = "hotpot.png"
image = cv2.imread(image_path)

# 確保圖片成功讀取
if image is None:
    print("❌ 無法讀取圖片，請確認圖片檔案是否存在。")
    exit()

# 調整圖片大小 (300x300)
input_shape = input_details[0]['shape']
image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

# 送入模型執行偵測
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 取得模型輸出
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]

# 解析並標記圖片上的物件
h, w, _ = image.shape
detected_objects = []
for i in range(len(scores)):
    if scores[i] > CONFIDENCE_THRESHOLD:  
        class_id = int(classes[i])
        class_name = labels[class_id].lower()

        # 🔴 只偵測 `ALLOWED_CLASSES` 內的物件
        if class_name not in ALLOWED_CLASSES:
            continue

        y_min, x_min, y_max, x_max = boxes[i]
        x_min, x_max = int(x_min * w), int(x_max * w)
        y_min, y_max = int(y_min * h), int(y_max * h)

        # 繪製邊界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 標示標籤與信心度
        label = f"{class_name}: {scores[i]:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detected_objects.append((class_name, scores[i]))

# 印出模型輸出( 驗收2 用)
print_model_output = True #是否印出模型輸出 (是 : 設為True, 否 : 設為False)

if print_model_output == True :
    print("以下為 boxes 的 output ( 4個1組 ) :")
    print(boxes)
    print('\n')

    print("以下為 classes 的 output ( 1個1組 ) :")
    print(classes)
    print('\n')

    print("以下為 scores 的 output ( 1個1組 ) :")
    print(scores)
    print('\n')

# =======驗收2部分=======

object_number = 0 # 此變數用於儲存特定種類物品的數量
scores_list = [] # 此變數用於儲存每個物件的信心度

# =======請在此範圍內撰寫你的code=======


# =======請在此範圍內撰寫你的code=======

print("這張圖片中偵測到" + str(object_number) + "個")
print("其個別信心度如下:")
print(scores_list)

# =======驗收2部分=======

# 顯示結果
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 終端機顯示所有偵測結果
print("\n🎯 偵測到的物體：")
for obj, conf in detected_objects:
    print(f"{obj} - 信心度: {conf:.2f}")

if not detected_objects:
    print("❌ 未偵測到符合條件的物體")
