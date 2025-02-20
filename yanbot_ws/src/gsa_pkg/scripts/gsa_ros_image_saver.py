#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
import time
from cv_bridge import CvBridge
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.save_dir = '/tmp/file_pipe/'
        self.flag_file = os.path.join(self.save_dir, 'image.flag')
        self.latest_image = None  # 存储最新的图像

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

    def image_callback(self, msg):
        try:
            # 仅更新最新图像，不进行保存操作
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error converting image: {str(e)}")

    def timer_callback(self, event):
        if self.latest_image is None:
            return

        try:
            temp_path = os.path.join(self.save_dir, "latest_image_tmp.jpg")
            final_path = os.path.join(self.save_dir, "latest_image.jpg")

            max_retries = 10  # 最大重试次数
            for _ in range(max_retries):
                flag_status = self.check_flag()
                if flag_status == 'reading':
                    time.sleep(0.001)
                    continue
                elif flag_status in ['writed', 'read', None]:
                    self.set_flag('writing')
                    cv2.imwrite(temp_path, self.latest_image)
                    os.rename(temp_path, final_path)
                    self.set_flag('writed')
                    break
                else:
                    self.set_flag('read')
                    time.sleep(0.001)
            else:
                rospy.logwarn("Failed to save image after max retries")

        except Exception as e:
            rospy.logerr(f"Error saving image: {str(e)}")
            self.set_flag('read')

    def start(self):
        rospy.init_node('image_saver', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Image, self.image_callback)
        # 创建10Hz定时器
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.spin()

if __name__ == '__main__':
    image_saver = ImageSaver()
    image_saver.start()