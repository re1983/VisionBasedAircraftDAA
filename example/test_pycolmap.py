import pycolmap

# 定義工作區路徑和資料夾
workspace = "project_directory"

# 進行增量式稀疏重建
reconstruction = pycolmap.Reconstruction.incremental_mapper(workspace)

# 保存結果到 sparse 目錄
reconstruction.export(path=f"{workspace}/sparse")
