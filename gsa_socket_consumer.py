#!/usr/bin/env python

import os
import socket
import cv2
import numpy as np
import time

class ImageReceiver:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', 7777))  # 绑定到容器的端口
        self.sock.listen(1)
        self.conn, self.addr = None, None

        print("Waiting for connection...")
        while self.conn is None:
            try:
                # 阻塞直到接收到连接
                self.conn, self.addr = self.sock.accept()
                print(f"Connection from {self.addr} established.")
            except Exception as e:
                print(f"Error during connection: {e}")
                time.sleep(1)  # 等待一段时间再尝试

    def receive_and_process(self):
        while True:
            # 接收图像数据
            data = self.recv_data()
            if not data:
                print("No data received, continuing to wait.")
                time.sleep(1)
                continue  # 如果没有数据，则等待

            # 转换为 NumPy 数组并解码为 OpenCV 图像
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("Failed to decode image.")
                continue  # 如果图像为空，继续等待

            # gray处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 将gray图像保存为文件
            cv2.imwrite("gray.jpg", gray)

    def recv_data(self):
        data = b''
        while len(data) < 1024 * 1024:  # 读取一小段数据
            packet = self.conn.recv(1024)
            if not packet:
                break
            data += packet
        return data

if __name__ == '__main__':
    receiver = ImageReceiver()
    receiver.receive_and_process()
