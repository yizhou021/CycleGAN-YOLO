# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ultralytics import YOLO, solutions

import torch
import os
from PIL import Image
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load a model
    print(torch.__version__);
    print(torch.cuda.is_available())


    model = YOLO('E:/project/yolov8_test/runs/train14-5mGAN/weights/best.pt')  # load a pretrained model (recommended for training)

    # model = YOLO('E:/project/yolov8_test/model/best.pt')



    # Train the model
    # results = model.train(data='E:/project/yolov8_test/dataset/coco8.yaml', epochs=500, imgsz=640,batch=6)

    # results = model.val()

    # Run batched inference on a list of images
    folder_path = "E:/project/yolov8_test/dataset/水葫芦统计/应用数据集"
    file_list = []

    # 使用os模块的listdir函数遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if 'jpg' in filename:
        # 拼接文件的完整路径
            file_path = folder_path+"/" + filename
        # 判断路径是否为文件
            if os.path.isfile(file_path):
                file_list.append(file_path)

    # results = model(['E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/test/000188.jpg', 'E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/test/000354.jpg','E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/test/000400.jpg','E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/test/000499.jpg'],iou =0.1)
    # return a list of Results objects
    results = model(file_list)

    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy arrayG
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image


        # Show results to screen (in supported environments)
        # r.show()
        # Save results to disk
        r.save(filename=f'E:/notebook/mining/yolo/result/5mGANbest_results{i}.jpg')


        model.export()

        # Save results to disk
    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs

    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     result.show()  # display to screen
    #     result.save(filename='E:/notebook/mining/yolo/result/result.jpg')  # save to disk


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
