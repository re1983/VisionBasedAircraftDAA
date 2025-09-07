import cv2
import numpy as np
import os
import glob
from pathlib import Path

def merge_images_vertically(image_folder, output_filename="merged_image.jpg", file_pattern="*.jpg"):
    """
    将指定文件夹中的图片按照文件名顺序从上到下拼接合并
    
    Args:
        image_folder (str): 图片文件夹路径
        output_filename (str): 输出文件名
        file_pattern (str): 文件匹配模式，如 "*.jpg", "*.png" 等
    """
    
    # 获取所有符合条件的图片文件
    image_files = glob.glob(os.path.join(image_folder, file_pattern))
    
    # 如果没有找到图片文件，尝试其他常见格式
    if not image_files:
        for ext in ["*.png", "*.jpeg", "*.bmp", "*.tiff"]:
            image_files = glob.glob(os.path.join(image_folder, ext))
            if image_files:
                break
    
    if not image_files:
        print(f"在文件夹 {image_folder} 中没有找到图片文件")
        return None
    
    # 按文件名排序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图片:")
    for i, img_file in enumerate(image_files):
        print(f"{i+1}. {os.path.basename(img_file)}")
    
    # 读取所有图片
    images = []
    max_width = 0
    total_height = 0
    
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        images.append(img)
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        total_height += height
        
        print(f"图片 {os.path.basename(img_path)}: {width}x{height}")
    
    if not images:
        print("没有成功读取任何图片")
        return None
    
    # 创建空白画布
    merged_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    merged_image.fill(255)  # 白色背景
    
    # 拼接图片
    current_y = 0
    for i, img in enumerate(images):
        height, width = img.shape[:2]
        
        # 计算居中位置
        x_offset = (max_width - width) // 2
        
        # 将图片放置到合并画布上
        merged_image[current_y:current_y + height, x_offset:x_offset + width] = img
        current_y += height
        
        print(f"已拼接第 {i+1} 张图片")
    
    # 保存合并后的图片
    output_path = os.path.join(image_folder, output_filename)
    success = cv2.imwrite(output_path, merged_image)
    
    if success:
        print(f"\n合并完成！")
        print(f"输出文件: {output_path}")
        print(f"最终尺寸: {max_width}x{total_height}")
        return output_path
    else:
        print("保存失败！")
        return None

def merge_images_with_custom_order(image_paths, output_filename="merged_custom.jpg"):
    """
    按照指定的图片路径列表顺序拼接图片
    
    Args:
        image_paths (list): 图片路径列表
        output_filename (str): 输出文件名
    """
    
    if not image_paths:
        print("图片路径列表为空")
        return None
    
    print(f"按照指定顺序拼接 {len(image_paths)} 张图片:")
    for i, img_path in enumerate(image_paths):
        print(f"{i+1}. {os.path.basename(img_path)}")
    
    # 读取所有图片
    images = []
    max_width = 0
    total_height = 0
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        images.append(img)
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        total_height += height
    
    if not images:
        print("没有成功读取任何图片")
        return None
    
    # 创建空白画布
    merged_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    merged_image.fill(255)  # 白色背景
    
    # 拼接图片
    current_y = 0
    for i, img in enumerate(images):
        height, width = img.shape[:2]
        
        # 计算居中位置
        x_offset = (max_width - width) // 2
        
        # 将图片放置到合并画布上
        merged_image[current_y:current_y + height, x_offset:x_offset + width] = img
        current_y += height
    
    # 保存合并后的图片
    folder = os.path.dirname(image_paths[0]) if image_paths else "."
    output_path = os.path.join(folder, output_filename)
    success = cv2.imwrite(output_path, merged_image)
    
    if success:
        print(f"\n合并完成！")
        print(f"输出文件: {output_path}")
        print(f"最终尺寸: {max_width}x{total_height}")
        return output_path
    else:
        print("保存失败！")
        return None

def preview_merged_image(output_path):
    """
    预览合并后的图片
    """
    if not os.path.exists(output_path):
        print(f"文件不存在: {output_path}")
        return
    
    img = cv2.imread(output_path)
    if img is None:
        print("无法读取合并后的图片")
        return
    
    # 调整显示尺寸（如果图片太大）
    height, width = img.shape[:2]
    max_display_height = 800
    
    if height > max_display_height:
        scale = max_display_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(img, (new_width, new_height))
    else:
        img_resized = img
    
    cv2.imshow('Merged Image Preview', img_resized)
    print("按任意键关闭预览窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    主函数 - 提供交互式界面
    """
    print("=== 图片垂直拼接工具 ===\n")
    
    # 获取当前脚本所在目录（pic文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前工作目录: {current_dir}")
    
    while True:
        print("\n请选择操作模式:")
        print("1. 自动拼接当前文件夹中的所有图片（按文件名排序）")
        print("2. 自动拼接指定文件夹中的所有图片")
        print("3. 手动指定图片路径和顺序")
        print("4. 快速拼接当前目录的PNG图片")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            # 自动拼接当前目录
            file_pattern = input("请输入文件匹配模式 (默认 *.png): ").strip()
            if not file_pattern:
                file_pattern = "*.png"
            
            output_name = input("请输入输出文件名 (默认 merged_image.jpg): ").strip()
            if not output_name:
                output_name = "merged_image.jpg"
            
            result = merge_images_vertically(current_dir, output_name, file_pattern)
            
            if result:
                preview_choice = input("是否预览合并结果？(y/n): ").strip().lower()
                if preview_choice == 'y':
                    preview_merged_image(result)
        
        elif choice == "2":
            # 自动模式 - 指定文件夹
            folder_path = input("请输入图片文件夹路径: ").strip()
            if not os.path.exists(folder_path):
                print("文件夹不存在！")
                continue
            
            file_pattern = input("请输入文件匹配模式 (默认 *.jpg): ").strip()
            if not file_pattern:
                file_pattern = "*.jpg"
            
            output_name = input("请输入输出文件名 (默认 merged_image.jpg): ").strip()
            if not output_name:
                output_name = "merged_image.jpg"
            
            result = merge_images_vertically(folder_path, output_name, file_pattern)
            
            if result:
                preview_choice = input("是否预览合并结果？(y/n): ").strip().lower()
                if preview_choice == 'y':
                    preview_merged_image(result)
        
        elif choice == "3":
            # 手动模式
            print("请输入图片路径（每行一个，输入空行结束）:")
            image_paths = []
            while True:
                path = input().strip()
                if not path:
                    break
                image_paths.append(path)
            
            if not image_paths:
                print("没有输入任何图片路径！")
                continue
            
            output_name = input("请输入输出文件名 (默认 merged_custom.jpg): ").strip()
            if not output_name:
                output_name = "merged_custom.jpg"
            
            result = merge_images_with_custom_order(image_paths, output_name)
            
            if result:
                preview_choice = input("是否预览合并结果？(y/n): ").strip().lower()
                if preview_choice == 'y':
                    preview_merged_image(result)
        
        elif choice == "4":
            # 快速拼接当前目录的PNG图片
            print("快速拼接当前目录的所有PNG图片...")
            result = merge_images_vertically(current_dir, "merged_result.jpg", "*.png")
            
            if result:
                preview_choice = input("是否预览合并结果？(y/n): ").strip().lower()
                if preview_choice == 'y':
                    preview_merged_image(result)
        
        elif choice == "5":
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入！")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    # 如果只是想快速合并当前目录的图片，可以取消注释下面的代码
    # merge_images_vertically(".", "merged_result.jpg", "*.png")
    
    # 运行交互式程序
    main()
