import torch
from pathlib import Path
import cv2
import numpy as np
import os



# YOLOv5 모델 로드
# model = torch.hub.load("../yolov5", 'custom',"runs/detect/exp4",source="local")
model = torch.hub.load(".", 'custom',"yolov5s",source="local")

# 개만을 감지할 클래스
class_names = ['dog']

# 클래스 목록을 모델에 설정
model.model.names = class_names

#루트 폴더
root_dir = os.getcwd()

# 동영상 파일 경로
video_path = f'{root_dir}/input/videos/dog_sample.mp4'

# 결과를 저장할 디렉토리
output_dir = f'{root_dir}/output'

# 디렉토리 생성
Path(output_dir+'/images/trainee2307').mkdir(parents=True, exist_ok=True)
Path(output_dir+'/labels/trainee2307').mkdir(parents=True, exist_ok=True)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,(640,640))

    # YOLOv5를 통해 개 객체 감지 수행
    results = model(frame)

    # 바운딩 박스 좌표와 라벨 추출
    bboxes = results.pred[0][:, :4].detach().cpu().numpy()
    labels = results.pred[0][:, -1].detach().cpu().numpy()

    # 개 객체만 추출
    dog_bboxes = []
    for bbox, label in zip(bboxes, labels):
        #기존 모델에서 개의 라벨 index는 16
        if label == 16:
            dog_bboxes.append(bbox/640)
            

    # 바운딩 박스 좌표를 파일에 저장
    output_file = f'{output_dir}/labels/trainee2307/frame_{frame_count}.txt'
    output_img = f'{output_dir}/images/trainee2307/frame_{frame_count}.jpg'
    with open(output_file, 'w') as f:
        for bbox in dog_bboxes:
            f.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')

    cv2.imwrite(output_img,frame)
    frame_count += 1

# 작업 완료 후, 캡처 해제
cap.release()
