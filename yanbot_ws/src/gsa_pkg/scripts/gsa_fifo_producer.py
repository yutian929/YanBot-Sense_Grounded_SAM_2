#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import struct
import os

class GSAFIFOProducer:
    def __init__(self):
        rospy.init_node('gsa_fifo_producer', anonymous=True)
        
        # 获取ROS参数
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.pipe_path = rospy.get_param('~pipe_path', '/tmp/fifo_pipe')
        
        # 创建命名管道
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)
            rospy.loginfo("创建命名管道: %s", self.pipe_path)
        else:
            rospy.loginfo("使用已有管道: %s", self.pipe_path)
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 订阅图像话题
        self.image_sub = rospy.Subscriber(self.rgb_topic, Image, self.image_callback)

    def image_callback(self, msg):
        try:
            # 转换ROS图像消息到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("图像转换错误: %s", e)
            return

        # 将图像编码为JPEG
        ret, jpeg_buffer = cv2.imencode('.jpg', cv_image)
        if not ret:
            rospy.logerr("JPEG编码失败")
            return
            
        jpeg_bytes = jpeg_buffer.tobytes()
        
        # 构造数据包（4字节长度头 + 图像数据）
        data = struct.pack('>I', len(jpeg_bytes)) + jpeg_bytes
        
        # 写入命名管道
        try:
            with open(self.pipe_path, 'wb') as pipe:
                pipe.write(data)
                pipe.flush()
        except IOError as e:
            rospy.logerr("管道写入错误: %s", e)

if __name__ == '__main__':
    try:
        producer = GSAFIFOProducer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass