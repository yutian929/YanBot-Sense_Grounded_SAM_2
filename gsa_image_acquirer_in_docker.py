#!/usr/bin/env python
import cv2
import numpy as np
import os
import time

class ImageAcquirer:
    def __init__(self, watch_dir='/tmp/file_pipe', flag_file='image.flag', image_file='latest_image.jpg'):
        self.watch_dir = watch_dir  # 监视的图像目录
        self.flag_file = os.path.join(self.watch_dir, flag_file)  # 标志文件路径
        self.image_path = os.path.join(self.watch_dir, image_file)

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
    
    def getting_image(self):
        while True:
            flag_status = self.check_flag()
            if flag_status in ('writing', 'read'):
                time.sleep(0.01)
                continue
            elif flag_status == 'writed':  # 可以开始读取
                self.set_flag('reading')  # 设置为读取状态
                # 验证文件完整性
                if self.check_jpeg_end():
                    img = self.read_image(self.image_path)
                    return self.image_path, img
                else:
                    self.set_flag("read")
            else:
                self.set_flag("read")  # 重新开始
                time.sleep(0.01)

            
    def after_getting_image(self):
        self.set_flag('read')

    def check_jpeg_end(self):
        retry = 3
        while retry > 0:
            try:
                with open(self.image_path, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() == b'\xff\xd9':  # JPEG结束标记
                        return True
            except:
                retry -= 1
                time.sleep(0.1)
        return False

    def watch_directory(self):
        while True:
            try:
                while True:
                    flag_status = self.check_flag()
                    if flag_status in ('writing', 'read'):
                        time.sleep(0.01)  # 添加等待
                        continue
                    elif flag_status == 'writed':
                        self.set_flag('reading')
                        # 验证文件完整性
                        if not self.check_jpeg_end():
                            self.set_flag('read')
                            continue
                        # 处理图像
                        img = self.read_image(self.image_path)
                        if img is not None:
                            self.process_image(img)
                        self.set_flag('read')
                        break
                    else:
                        self.set_flag('read')
                        time.sleep(0.1)
            
            except Exception as e:
                print(f"Processing error: {str(e)}")
                self.set_flag('read')  # 异常时重置状态

if __name__ == '__main__':
    processor = ImageAcquirer()
    processor.watch_directory()



