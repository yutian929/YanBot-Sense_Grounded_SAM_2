#!/usr/bin/env python
import cv2
import numpy as np
import os
import time

class ImageProcessor:
    def __init__(self):
        self.watch_dir = '/tmp/file_pipe'  # 监视的图像目录
        self.flag_file = os.path.join(self.watch_dir, 'image.flag')  # 标志文件路径
        self.last_image_time = 0

    def check_flag(self):
        """检查 flag 文件的状态，确保主机没有在写入"""
        if not os.path.exists(self.flag_file):
            return None
        with open(self.flag_file, 'r') as flag_file:
            return flag_file.read().strip()

    def set_flag(self, status):
        """设置 flag 文件的状态"""
        with open(self.flag_file, 'w') as flag_file:
            flag_file.write(status)

    def read_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image from {image_path}")
            return
        return img
    def process_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray.png", gray)
        time.sleep(1)
        print(f"Processed image saved as gray.png.")

    def watch_directory(self):
        while True:
            # 1. 确保主机端不在写入
            while True:
                flag_status = self.check_flag()
                if flag_status == 'writing' or flag_status == 'read':  # 如果主机在写入或读取过了，等待
                    print("writing / read")
                    continue
                elif flag_status == 'writed':  # 可以开始读取
                    self.set_flag('reading')  # 设置为读取状态
                    print("reading")
                    break
                else:
                    self.set_flag("read")  # 重新开始
                    print("restart")

            image_path = os.path.join(self.watch_dir, 'latest_image.jpg')

            if os.path.exists(image_path):
                last_modified_time = os.path.getmtime(image_path)

                if last_modified_time > self.last_image_time:
                    print(f"New image detected: {image_path}")
                    
                    # 2. 读取并处理图像
                    img = self.read_image(image_path)
                    self.set_flag('read')

                    self.process_image(img)
                    self.last_image_time = last_modified_time

            # time.sleep(1)  # 每秒钟检查一次

if __name__ == '__main__':
    processor = ImageProcessor()
    processor.watch_directory()



