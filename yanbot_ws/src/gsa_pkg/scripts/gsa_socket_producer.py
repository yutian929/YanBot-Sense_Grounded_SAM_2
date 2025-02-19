#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import time
from sensor_msgs.msg import Image
import cv2
import socket
import numpy as np
from cv_bridge import CvBridge

class ImagePublisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 确保容器端准备好
        connected = False
        while not connected:
            try:
                self.sock.connect(('localhost', 7777))  # 尝试连接到容器端
                connected = True
                print("Connected to the container.")
            except Exception as e:
                print(f"Waiting for container to be ready: {e}")
                time.sleep(1)  # 等待容器端准备好

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # 将图像编码为 JPEG
        ret, encoded_image = cv2.imencode('.jpg', cv_image)
        if ret:
            try:
                # 将图像发送到 Docker 容器
                self.sock.sendall(encoded_image.tobytes())
            except Exception as e:
                print(f"Error sending image: {e}")

    def start(self):
        rospy.init_node('image_publisher', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Image, self.image_callback)
        rospy.spin()

if __name__ == '__main__':
    image_publisher = ImagePublisher()
    image_publisher.start()
