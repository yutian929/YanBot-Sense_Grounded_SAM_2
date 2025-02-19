#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.save_dir = '/tmp/file_pipe/'  # 保存图像的目录
        self.flag_file = os.path.join(self.save_dir, 'image.flag')  # 标志文件路径

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def check_flag(self):
        """检查 flag 文件的状态，确保容器没有在读取图像"""
        if not os.path.exists(self.flag_file):
            return None
        with open(self.flag_file, 'r') as flag_file:
            return flag_file.read().strip()

    def set_flag(self, status):
        """设置 flag 文件的状态"""
        with open(self.flag_file, 'w') as flag_file:
            flag_file.write(status)

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_path = os.path.join(self.save_dir, "latest_image.jpg")

        # 1. 确保容器端不在读取
        while True:
            flag_status = self.check_flag()
            if flag_status == 'reading':  # 如果容器在读取，等待
                print("reading")
                continue
            elif flag_status == 'writed' or flag_status == 'read':  # 可以开始写入
                self.set_flag('writing')  # 设置为写入状态
                print("writing")
                break
            else:
                self.set_flag('read') # 重新开始
                print("restart")

        # 2. 写入图像文件
        cv2.imwrite(image_path, cv_image)
        print(f"Image saved at {image_path}")

        # 3. 写入后，标记为 writed 状态，表示图像已经写入
        self.set_flag('writed')
        print(f"Image processing completed. Flag set to 'writed'.")

    def start(self):
        rospy.init_node('image_saver', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Image, self.image_callback)
        rospy.spin()

if __name__ == '__main__':
    image_saver = ImageSaver()
    image_saver.start()


