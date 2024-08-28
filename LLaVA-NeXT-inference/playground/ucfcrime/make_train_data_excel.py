import os

# 设置根目录路径
root_dir = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train"
# 输出文件路径
output_file = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/ucf_crime_videos.txt"

# 用于存储所有视频文件的相对路径
video_paths = []

# 获取根目录下的直接子目录
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):  # 确保它是一个目录
        # 获取子目录下的所有文件
        for file in os.listdir(subdir_path):
            # 检查文件扩展名是否为常见的视频格式
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # 构建 */ *.* 的路径格式
                relative_path = os.path.join(subdir, file)
                # 将路径添加到列表中
                video_paths.append(relative_path)

# 对所有视频文件路径进行全局排序
video_paths.sort()

# 将排序后的路径写入输出文件
with open(output_file, 'w') as f:
    for path in video_paths:
        f.write(path + '\n')

print(f"视频路径已保存到 {output_file}")


import pandas as pd

# 输入文件路径
input_file = output_file
# 输出 Excel 文件路径
output_file = "//mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/ucf_crime_videos.xlsx"

# 读取文件内容，每一行作为一个元素存储在列表中
with open(input_file, 'r') as f:
    video_paths = [line.strip() for line in f]

# 创建一个 DataFrame，将每个路径放在单独的一行
df = pd.DataFrame(video_paths, columns=["Video Path"])

# 将 DataFrame 保存为 Excel 文件
df.to_excel(output_file, index=False)

print(f"视频路径已保存为 Excel 文件：{output_file}")
