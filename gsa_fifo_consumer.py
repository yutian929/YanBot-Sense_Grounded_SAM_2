#!/usr/bin/env python

import cv2
import struct
import os
import numpy as np

def read_exactly(fd, n):
    data = b''
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def main():
    pipe_path = '/tmp/fifo_pipe'  # 必须与生产者路径一致
    
    if not os.path.exists(pipe_path):
        print(f"错误: 管道 {pipe_path} 不存在")
        return

    fd = os.open(pipe_path, os.O_RDONLY)
    try:
        while True:
            # 读取4字节长度头
            header = read_exactly(fd, 4)
            if not header:
                print("管道已关闭")
                break
                
            # 解析图像数据长度
            length = struct.unpack('>I', header)[0]
            
            # 读取图像数据
            data = read_exactly(fd, length)
            if not data:
                print("不完整的数据包")
                break
                
            # 解码图像
            img_np = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # cv2.imshow('GSA FIFO Consumer', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # breakpoint()
                cv2.imwrite("frame.jpg", frame)
                print("111")
            else:
                print("图像解码失败")
    finally:
        os.close(fd)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()