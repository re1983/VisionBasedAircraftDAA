import sqlite3
from pathlib import Path

project_directory = Path('project_directory')
image_dir = project_directory / 'test'  # 图片存储的文件夹
database_path = project_directory / 'database.db'  # 创建一个数据库文件
sparse_output_path = project_directory / 'sparse'  # 存储稀疏结果的文件夹
input_path = project_directory / 'input'  # 包含 images.txt 和 cameras.txt 的文件夹

try:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # 检查 descriptors 表
    cursor.execute("SELECT COUNT(*) FROM descriptors")
    descriptor_count = cursor.fetchone()[0]
    print(f"Descriptors count: {descriptor_count}")
    
    # 检查 images 表
    cursor.execute("SELECT COUNT(*) FROM images")
    images_count = cursor.fetchone()[0]
    print(f"Images count: {images_count}")
    
    # 检查 matches 表
    cursor.execute("SELECT COUNT(*) FROM matches")
    matches_count = cursor.fetchone()[0]
    print(f"Matches count: {matches_count}")
    
    conn.close()
except Exception as e:
    print(f"Error: {str(e)}")