import os
import csv

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 遍历目录下所有文件，筛选PDB文件并提取PDB ID
pdb_ids = []
for filename in os.listdir(current_dir):
    # 检查文件是否以.pdb结尾（不区分大小写）
    if filename.lower().endswith(".pdb"):
        # 提取PDB ID（文件名前缀，假设PDB ID为文件名前4位，如1abc.pdb → 1ABC）
        pdb_id = filename[:6]  # 转换为大写格式
        pdb_ids.append({"PDB_ID": pdb_id})

# 保存为CSV文件
csv_path = os.path.join(current_dir, "pdb_ids.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["PDB_ID"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入表头和数据
    writer.writeheader()
    writer.writerows(pdb_ids)

print(f"成功提取 {len(pdb_ids)} 个PDB ID，保存至 {csv_path}")