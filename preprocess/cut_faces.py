from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import torch
import alignment

mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

img = Image.open("image.png")

boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

if boxes is None:
    print("Không tìm thấy mặt")
else:
    box = boxes[0]   
    conf = probs[0] 
    lm = landmarks[0]
    
    left_eye = lm[0]
    right_eye = lm[1]

    print("Confidence:", conf)
    print("Left eye :", left_eye)
    print("Right eye:", right_eye)

    # Crop ảnh
    x1, y1, x2, y2 = map(int, box)
    img_cv = cv2.imread("image.png")
    face_crop = img_cv[y1:y2, x1:x2]
    cv2.imwrite("crop.jpg", face_crop)
    align_face = alignment.alignment_procedure(face_crop, left_eye, right_eye)
    cv2.imwrite("aligned.jpg", face_crop)
