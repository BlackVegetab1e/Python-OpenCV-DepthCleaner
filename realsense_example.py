import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch

import PyDepthInpaint

 
class depth_camera_connect(threading.Thread):   #继承父类threading.Thread
    def __init__(self, depth_wide = 848, depth_height = 480, depth_fps = 30, depth_video = True):
        threading.Thread.__init__(self)
        # 848 480 is the optimal resolution

        self.setDaemon(True) # 设置为守护线程，当主线程退出时，守护线程也会退出。
        # Configure depth streams
        self.pipeline = rs.pipeline()
        self.decimation_filter = rs.decimation_filter()   # 下采样 降低分辨率
        self.decimation_filter.set_option(rs.option.filter_magnitude, 8)
        config = rs.config()
        config.enable_stream(rs.stream.depth, depth_wide, depth_height, rs.format.z16, depth_fps)
        profile = self.pipeline.start(config)
        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        # 设置激光功率为最大值 MAX POWER
        depth_sensor.set_option(rs.option.laser_power, 360)

        self.depth_cleaner = PyDepthInpaint.DepthProcess(108,60)

        self.depth_video = depth_video
        if(self.depth_video):
            import datetime
            self.output_video = './videos/depth'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))+'.mp4'

            frame_width = 87
            frame_height = 58

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
            self.out = cv2.VideoWriter(self.output_video, fourcc, depth_fps, (frame_width, frame_height), False)  # False表示灰度图


        # 开启线程
        self.start()
        self.tlast = time.time()
        self.camera_time = None
        self.running = True  # 标志线程是否在运行



    def run(self):          
        while True:
            
            if not self.running:
                break
            # Wait for frames: depth
            frames = self.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # 下采样，滤波处理
            depth_frame = self.decimation_filter.process(depth_frame)
            depth_image = np.asarray(depth_frame.get_data(), dtype=np.uint16)
            depth_image = self.depth_cleaner.process(depth_image)

            depth_np = self.process_depth_image(depth_image/1000.0)
            if(self.depth_video): 
                # cv2.imwrite("./depth.png", (depth_np+0.5)*255)
                self.out.write(np.uint8((depth_np+0.5)*255))
            self.depth_torch = torch.tensor(depth_np,device="cuda").unsqueeze(0).float()

            self.tlast =time.time()


    def shutdown_rs(self):
        if self.running:
            self.running = False  # 设置线程停止标志
            self.pipeline.stop()  # 停止 RealSense 流
            if(self.depth_video): 
                self.out.release()
                print("video save at",self.output_video)
            print("real_sense Pipeline 已停止, 相机正常关闭")
        else:
            print("相机已经关闭")

    


    
    def normalize_depth_image(self, depth_image):
        depth_image = (depth_image - 0.3) / (3 - 0.3) - 0.5
        return depth_image

    def process_depth_image(self, depth_image):
        depth_image = np.clip(depth_image, 0.3, 5)
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image



if __name__ == "__main__":
    depth_camera = depth_camera_connect()
    time.sleep(10)
    depth_camera.shutdown_rs()
