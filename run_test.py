import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
import argparse

# 人脸特征提取函数
def extract_face_encoding(img_path):
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用OpenCV的DNN人脸检测器
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    if len(detections) > 0:
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = rgb[startY:endY, startX:endX]
        
        # 简化版特征提取（实际应使用FaceNet等模型）
        return cv2.resize(face, (128, 128)).flatten()
    return None

# 人脸识别函数
def recognize_face(test_encoding, known_faces):
    if test_encoding is None:
        return "未知", 0
    
    min_dist = float('inf')
    best_match = None
    
    for (name, enc) in known_faces:
        if enc is not None:
            dist = np.linalg.norm(enc - test_encoding)
            if dist < min_dist:
                min_dist = dist
                best_match = name
                
    return best_match if min_dist < 0.6 else "未知", 1 - min_dist

# 生成报告
def generate_report(results):
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results)
    
    print(f"\n测试结果汇总：")
    print(f"总样本数: {len(results)}")
    print(f"正确识别: {correct}")
    print(f"准确率: {accuracy:.2%}")
    
    # 保存详细结果
    with open("test_results.txt", "w") as f:
        for r in results:
            f.write(f"{r['file']}\t{r['true']}\t{r['pred']}\t{r['correct']}\t{r['confidence']:.2f}\n")

def batch_test(test_dir, db_dir):
    known_faces = []
    
    # 加载已知人脸
    print("正在加载已注册人脸...")
    for person_dir in tqdm(os.listdir(db_dir)):
        person_path = os.path.join(db_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                encoding = extract_face_encoding(img_path)
                known_faces.append((person_dir, encoding))
    
    # 执行测试
    results = []
    print("\n开始批量测试...")
    for img_file in tqdm(os.listdir(test_dir)):
        test_path = os.path.join(test_dir, img_file)
        true_name = os.path.splitext(img_file)[0].split("_")[0]
        
        test_encoding = extract_face_encoding(test_path)
        pred_name, confidence = recognize_face(test_encoding, known_faces)
        
        results.append({
            "file": img_file,
            "true": true_name,
            "pred": pred_name,
            "correct": true_name == pred_name,
            "confidence": confidence
        })
    
    generate_report(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="./test_dataset")
    parser.add_argument("--db_dir", default="./data/face_img_database")
    args = parser.parse_args()
    
    batch_test