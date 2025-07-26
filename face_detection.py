import cv2
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = r"D:\face-attend\WorkAttendanceSystem-master\V2.0"
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXCEL_REPORT = os.path.join(BASE_DIR, "detection_report.xlsx")

# 选择模型类型 ('haar' 或 'dnn')
MODEL_TYPE = 'dnn'  # 推荐使用dnn更准确

# ================= 模型初始化 =================
if MODEL_TYPE == 'haar':
    # Haar级联分类器
    model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(model_path)
else:
    # DNN模型（需下载两个文件）
    proto_path = os.path.join(BASE_DIR, "deploy.prototxt")
    model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# ================= 检测函数 =================
def detect_faces(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}")
        return 0

    if MODEL_TYPE == 'haar':
        # Haar检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    else:
        # DNN检测
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))

    # 绘制检测框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 保存结果
    output_path = os.path.join(RESULTS_DIR, f"detected_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, img)
    return len(faces)

# ================= 主流程 =================
if __name__ == "__main__":
    # 创建目录
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 获取图片列表
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print("="*50)
    print(f"测试配置：")
    print(f"模型类型：{'DNN' if MODEL_TYPE == 'dnn' else 'Haar'}")
    print(f"图片数量：{len(image_files)}")
    print("="*50)

    # 多线程处理
    with ThreadPoolExecutor() as executor:
        image_paths = [os.path.join(TEST_IMAGES_DIR, f) for f in image_files]
        results = list(tqdm(executor.map(detect_faces, image_paths), total=len(image_files)))

    # 生成报告
    report_data = {
        "图片名": image_files,
        "检测到人脸数": results,
        "检测方式": ["DNN" if MODEL_TYPE == 'dnn' else "Haar"] * len(image_files)
    }
    
    df = pd.DataFrame(report_data)
    df.to_excel(EXCEL_REPORT, index=False)
    
    print("\n" + "="*50)
    print(f"测试完成！结果已保存：")
    print(f"- 标记图片：{RESULTS_DIR}")
    print(f"- Excel报告：{EXCEL_REPORT}")
    print("="*50)