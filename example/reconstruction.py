import pycolmap
import pathlib

# 设置路径
project_dir = pathlib.Path("project_dir")
print(project_dir)
image_dir = project_dir / "images"
print(image_dir)
sparse_dir = project_dir / "sparse"
print(sparse_dir)
output_dir = project_dir / "output"
print(output_dir)
database_path = output_dir / "database.db"
print(database_path)

# 创建输出目录
output_dir.mkdir(exist_ok=True)

# 加载已有的重建
reconstruction = pycolmap.Reconstruction(sparse_dir)

# 提取特征
pycolmap.extract_features(database_path, image_dir)

# 特征匹配
pycolmap.match_exhaustive(database_path)

# 增量式重建
maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

# 保存重建结果
maps.write(output_dir)

# 打印重建摘要
print(reconstruction.summary())

# 可选：稠密重建
mvs_path = output_dir / "mvs"
pycolmap.undistort_images(mvs_path, output_dir, image_dir)
# 注意：patch_match_stereo 需要 CUDA 支持
# pycolmap.patch_match_stereo(mvs_path)