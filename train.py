import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/best2.pt')
    results = model.train(
        data = "yaml/food.yaml",    #指定訓練任務檔
        imgsz = 256,                    #輸入影像大小
        epochs = 100,                   #訓練世代數
        patience = 25,                  #等待世代數，無改善就提前結束訓練
        batch = 15,                     #批次大小
        project = 'yolov8_object_1',    #專案名稱
        name = 'exp01'        )         #訓練實驗名稱