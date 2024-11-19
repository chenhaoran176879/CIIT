import pandas as pd

# 定义文件路径和所需列名
file_paths = [
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data_8frames.csv",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data_16frames.csv",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data_32frames.csv",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data_64frames.csv"
]

# 定义模型的输出顺序
model_order = [
    "InternVL2-1B", "qwen-VL2-2B", "qwen-VL2-7B",
    "llava-ov-qwen7B", "InternVL2-8B", "InternVL2-26B", "InternVL2-40B"
]

# 初始化空的 DataFrame，用于存储合并后的数据
merged_df = pd.DataFrame()

for file_path in file_paths:
    # 读取文件并提取所需列
    df = pd.read_csv(file_path, usecols=["model_name", "sum"])
    
    # 根据文件名提取帧数并重命名列
    frames = file_path.split("_")[-1].replace(".csv", "")
    df = df.rename(columns={"sum": f"score_{frames}"})
    
    # 合并数据
    if merged_df.empty:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="model_name", how="outer")

# 使用 model_order 对 DataFrame 进行手动排序
# 将模型名称按给定顺序映射为一个排序用的临时列
merged_df['order'] = merged_df['model_name'].apply(lambda x: model_order.index(x) if x in model_order else -1)

# 根据临时列进行排序
merged_df = merged_df.sort_values('order').drop('order', axis=1).reset_index(drop=True)

# 输出结果
print(merged_df)
merged_df.to_csv("/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data_nframes.csv")
