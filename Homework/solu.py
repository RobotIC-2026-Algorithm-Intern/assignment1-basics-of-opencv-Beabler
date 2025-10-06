import cv2
import numpy as np
import os
from pathlib import Path
import json


def load_colorconfig( config_path = "./config.json"):
    color_ranges = {}
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for color_name, ranges in config["color_ranges"].items():
        r = ranges[0]  
        lower = r["min"]
        upper = r["max"]
        color_ranges[color_name] = (lower, upper) 
    return color_ranges

def load_roiconfig( config_path = "./config.json"):
    roi_scale = []
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    figure = config["roi"]  
    
    y = figure["y"]
    yh = figure["y+h"]
    x = figure["x"]
    xw = figure["x+w"]  
    
    roi_scale = [y, yh, x, xw]
    return roi_scale



class ColorRange:
    def __init__(self):
        pass

    def generate_mask(self, color_ranges, hsv_image, color_name):
        if color_name != "red":
            lower, upper = color_ranges[color_name]
            mask = cv2.inRange(hsv_image,np.array(lower), np.array(upper))
        else:
            lower1,upper1 = color_ranges["red1"]
            lower2,upper2 = color_ranges["red2"]
            mask1 = cv2.inRange(hsv_image,np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_image,np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)

        return mask
    
    def generate_combined_mask(self, hsv_image, color_names, color_ranges):
        masks = []
        for color in color_names:
            mask = self.generate_mask(color_ranges, hsv_image, color)
            masks.append(mask)
        
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, m)
        return combined_mask


class BallDetector:
     def __init__(self, video_path):
        self.path = video_path
        
     def roi_process(self,frame):      
        scale = load_roiconfig()     
        y, yh, x, xw = scale     
        frame = frame[y:yh, x:xw]
        return frame



     def video_process(self):
            cap = cv2.VideoCapture(self.path)
            colorrange = ColorRange()

            # 检查视频是否成功打开
            if not cap.isOpened():
                print("Error: 无法打开视频文件。")
                exit()


            color_ranges = load_colorconfig()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps == float('inf'):
                fps = 30  # 假设默认帧率为 30 FPS
            print(f"视频帧率: {fps} FPS")


            delay = int(1000 / fps)
            print(f"每帧延迟: {delay} 毫秒")

            frame_count = 0
            state = "无"
            position = (50,100)
            front = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            color = (0, 255, 255)
            thickness = 3
            color_names = ["red","purple","blue"]




            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                
                lim_frame = self.roi_process(frame)

                img_hsv = cv2.cvtColor(lim_frame, cv2.COLOR_BGR2HSV_FULL)
                
                #获得组合掩膜
                combined_mask = colorrange.generate_combined_mask(img_hsv, color_names, color_ranges)
                num_nonzero = np.count_nonzero(combined_mask)
                if num_nonzero >= 6000:
                    mask_red = colorrange.generate_mask(color_ranges, img_hsv, color_name = "red")
                    red_pixel_count = cv2.countNonZero(mask_red)
                    mask_purple = colorrange.generate_mask(color_ranges, img_hsv, color_name = "purple")
                    purple_pixel_count = cv2.countNonZero(mask_purple)
                    mask_blue = colorrange.generate_mask(color_ranges, img_hsv, color_name = "blue")
                    blue_pixel_count = cv2.countNonZero(mask_blue)
                    color_counts = {'red': red_pixel_count, 'blue': blue_pixel_count, 'purple': purple_pixel_count}
                    state = max(color_counts, key=color_counts.get)

                else:
                    state = "no ball" 

                frame_count += 1

                
                #cmph_state = str(state)+str(num_nonzero)


                #cv2.putText(frame, cmph_state, position, front, font_scale, color, thickness)
                cv2.putText(frame, state, position, front, font_scale, color, thickness)
                cv2.imshow('Video Player', frame)

                if cv2.waitKey(delay) & 0xFF == ord('q'):  ######### delay
                    break  # 按 'q' 可提前退出
            # 释放视频捕获对象
            cap.release()
            cv2.destroyAllWindows()
            print(f"视频提取完成！共提取 {frame_count} 帧。")


def main():
    # 视频文件路径
    video_path1 = '../res/output.avi'  
    video_path2 = '../res/output1.avi' 
    balldetector = BallDetector(video_path1)
    balldetector.video_process()
    balldetector = BallDetector(video_path2)
    balldetector.video_process()

if __name__ == "__main__":
    main()




