#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import os

class ImageSaver:
    def __init__(self):
        self.save_dir = '/tmp/file_pipe/'
        self.flag_file = os.path.join(self.save_dir, 'image2.flag')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def check_flag(self):
        if not os.path.exists(self.flag_file):
            return None
        with open(self.flag_file, 'r') as flag_file:
            return flag_file.read().strip()

    def set_flag(self, status):
        with open(self.flag_file, 'w') as flag_file:
            flag_file.write(status)

    def save_image(self, image_cv):
        try:
            temp_path = os.path.join(self.save_dir, "latest_image2_tmp.jpg")
            final_path = os.path.join(self.save_dir, "latest_image2.jpg")

            max_retries = 10  # 最大重试次数
            for _ in range(max_retries):
                flag_status = self.check_flag()
                if flag_status == 'reading':
                    time.sleep(0.001)
                    continue
                elif flag_status in ['writed', 'read', None]:
                    self.set_flag('writing')
                    cv2.imwrite(temp_path, image_cv)
                    os.rename(temp_path, final_path)
                    self.set_flag('writed')
                    break
                else:
                    self.set_flag('read')
                    time.sleep(0.001)
            else:
                print("ERROR: Failed to save image after max retries")

        except Exception as e:
            print(f"ERROR: Error saving image: {str(e)}")
            self.set_flag('read')


if __name__ == '__main__':
    image_saver = ImageSaver()
    import numpy as np
    img_random = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_saver.write_image(img_random)