#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水平图片合并工具
将两张指定的图片左右合并
"""

import cv2
import numpy as np
import os

def merge_images_horizontally(left_image_path, right_image_path, output_path):
    """
    水平合并两张图片
    
    Args:
        left_image_path: 左侧图片路径
        right_image_path: 右侧图片路径
        output_path: 输出图片路径
    
    Returns:
        bool: 合并是否成功
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(left_image_path):
            print(f"错误: 左侧图片文件不存在: {left_image_path}")
            return False
            
        if not os.path.exists(right_image_path):
            print(f"错误: 右侧图片文件不存在: {right_image_path}")
            return False
        
        # 读取图片
        print(f"正在读取左侧图片: {left_image_path}")
        left_img = cv2.imread(left_image_path)
        if left_img is None:
            print(f"错误: 无法读取左侧图片: {left_image_path}")
            return False
            
        print(f"正在读取右侧图片: {right_image_path}")
        right_img = cv2.imread(right_image_path)
        if right_img is None:
            print(f"错误: 无法读取右侧图片: {right_image_path}")
            return False
        
        # 获取图片尺寸
        left_height, left_width = left_img.shape[:2]
        right_height, right_width = right_img.shape[:2]
        
        print(f"左侧图片尺寸: {left_width}x{left_height}")
        print(f"右侧图片尺寸: {right_width}x{right_height}")
        
        # 统一高度 - 使用较小的高度
        target_height = min(left_height, right_height)
        
        # 调整图片尺寸到统一高度，保持宽高比
        if left_height != target_height:
            scale_factor = target_height / left_height
            new_left_width = int(left_width * scale_factor)
            left_img = cv2.resize(left_img, (new_left_width, target_height))
            print(f"左侧图片调整为: {new_left_width}x{target_height}")
            left_width = new_left_width
            
        if right_height != target_height:
            scale_factor = target_height / right_height
            new_right_width = int(right_width * scale_factor)
            right_img = cv2.resize(right_img, (new_right_width, target_height))
            print(f"右侧图片调整为: {new_right_width}x{target_height}")
            right_width = new_right_width
        
        # 水平合并图片
        print("正在合并图片...")
        merged_img = np.hstack([left_img, right_img])
        
        # 保存合并后的图片
        success = cv2.imwrite(output_path, merged_img)
        
        if success:
            final_height, final_width = merged_img.shape[:2]
            print(f"合并成功!")
            print(f"输出文件: {output_path}")
            print(f"最终尺寸: {final_width}x{final_height}")
            return True
        else:
            print(f"错误: 保存图片失败: {output_path}")
            return False
            
    except Exception as e:
        print(f"合并过程中发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== 水平图片合并工具 ===\n")
    
    # 硬编码的文件路径
    left_image_path = "/home/re1983/extra/Work/VisionBasedAircraftDAA/pic/C_172_merged_image.jpg"
    right_image_path = "/home/re1983/extra/Work/VisionBasedAircraftDAA/pic/B737_800_merged_image.jpg"
    output_path = "/home/re1983/extra/Work/VisionBasedAircraftDAA/pic/horizontal_merged.jpg"
    
    print(f"左侧图片: {left_image_path}")
    print(f"右侧图片: {right_image_path}")
    print(f"输出文件: {output_path}")
    print("-" * 50)
    
    # 执行合并
    success = merge_images_horizontally(left_image_path, right_image_path, output_path)
    
    if success:
        print("\n✅ 水平合并完成!")
        
        # 询问是否预览
        try:
            preview = input("\n是否预览合并结果？(y/n): ").lower().strip()
            if preview in ['y', 'yes', '是']:
                print("正在打开预览...")
                # 读取并显示图片
                merged_img = cv2.imread(output_path)
                if merged_img is not None:
                    # 缩放图片以适应屏幕
                    height, width = merged_img.shape[:2]
                    if width > 1920:  # 如果图片太宽，缩放到1920像素宽
                        scale = 1920 / width
                        new_width = 1920
                        new_height = int(height * scale)
                        display_img = cv2.resize(merged_img, (new_width, new_height))
                    else:
                        display_img = merged_img
                    
                    cv2.imshow('Horizontal Merged Image', display_img)
                    print("按任意键关闭预览窗口...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("无法读取合并后的图片进行预览")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
    else:
        print("\n❌ 水平合并失败!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
