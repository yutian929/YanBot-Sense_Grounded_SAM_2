#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageProcessor:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_processor_node', anonymous=True)
        
        # 创建发布者和CvBridge
        self.publisher = rospy.Publisher('topic_processed_image', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # 文件监视配置
        self.watch_dir = rospy.get_param('~watch_dir', '/tmp/file_pipe')
        self.flag_file = os.path.join(self.watch_dir, 'image2.flag')
        self.image_file = os.path.join(self.watch_dir, 'latest_image2.jpg')
        
        # 设置10Hz定时器
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        # 初始化标志文件状态
        self.initialize_flag()

    def initialize_flag(self):
        """确保标志文件存在并初始化为read状态"""
        if not os.path.exists(self.flag_file):
            with open(self.flag_file, 'w') as f:
                f.write('read')

    def check_flag(self):
        """读取标志文件状态"""
        try:
            with open(self.flag_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            rospy.logerr(f"Error reading flag: {e}")
            return None

    def set_flag(self, status):
        """设置标志文件状态"""
        try:
            with open(self.flag_file, 'w') as f:
                f.write(status)
        except Exception as e:
            rospy.logerr(f"Error setting flag: {e}")

    def check_jpeg_end(self):
        """验证JPEG文件完整性"""
        try:
            with open(self.image_file, 'rb') as f:
                f.seek(-2, 2)
                return f.read() == b'\xff\xd9'
        except Exception as e:
            rospy.logerr(f"Integrity check failed: {e}")
            return False

    def process_and_publish(self):
        """处理图像并发布"""
        try:
            # 读取图像
            img = cv2.imread(self.image_file)
            if img is None:
                rospy.logwarn("Failed to read image file")
                return
            # 转换为ROS消息并发布
            msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.publisher.publish(msg)
            rospy.loginfo_once("Successfully published processed image")
            
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")

    def timer_callback(self, event):
        """定时器回调函数（10Hz）"""
        flag_status = self.check_flag()
        
        if flag_status == 'writed':
            # 设置读取状态
            self.set_flag('reading')
            
            # 验证文件完整性
            if self.check_jpeg_end():
                self.process_and_publish()
            else:
                rospy.logwarn("Invalid image file detected")
            
            # 重置为已读状态
            self.set_flag('read')
            
        elif flag_status in ('writing', 'reading'):
            # 等待当前操作完成
            pass
            
        else:
            # 处理异常状态
            self.set_flag('read')

    def shutdown_hook(self):
        """节点关闭时的清理工作"""
        self.timer.shutdown()
        rospy.loginfo("Node shutdown complete")

if __name__ == '__main__':
    try:
        processor = ImageProcessor()
        rospy.on_shutdown(processor.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass