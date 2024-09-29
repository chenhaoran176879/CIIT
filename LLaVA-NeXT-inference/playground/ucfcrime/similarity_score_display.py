import streamlit as st
import pandas as pd
import os

# 基础路径和CSV文件路径
BASE_DIR = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/"
score_csv = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/similarity_score_results.csv"
caption_csv = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/video_caption_comparison.csv"

# 读取CSV文件
@st.cache_data
def load_score_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

@st.cache_data
def load_caption_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# 查找视频路径
def find_video_path(video_name):
    category = video_name.split('_')[0][:-3]
    video_path = os.path.join(BASE_DIR, category, video_name)
    return video_path if os.path.exists(video_path) else None

# 加载视频文件和对应信息
def display_video_info(df, caption_df, selected_video):
    # 找到对应的视频路径
    video_path = find_video_path(selected_video)
    
    if video_path:
        # 在前端显示视频
        st.video(video_path)
        
        # 获取该视频的注释信息
        video_info = df[df['video_name'] == selected_video].iloc[0]
        st.write("**视频名称:**", video_info['video_name'])
        
        HUMAN_TIME = None
        LLM_TIME = None
        captions = caption_df[caption_df['video_name'] == selected_video]
        if not captions.empty:
            for _, row in captions.iterrows():
                if row['标注作者'] == '人工':
                    HUMAN_TIME = row
                elif row['标注作者'] == 'LLM':
                    LLM_TIME = row

        else:
            st.write("没有找到该视频的标注信息。")

        # 分栏显示信息
        col1, col2 = st.columns(2)
        with col1:
            st.write("**人工注释:**")
            st.write(video_info['human_description'])
            st.write(f"**开始时间:** {HUMAN_TIME['start_time']} s")
            st.write(f"**结束时间:** {HUMAN_TIME['end_time']} s")
        with col2:
            st.write("**AI注释:**")
            st.write(video_info['other_description'])
            st.write(f"**开始时间:** {LLM_TIME['start_time']} s")
            st.write(f"**结束时间:** {LLM_TIME['end_time']} s")
        
        

        # 显示相似度评分
        st.write("**相似度评分:**", video_info['similarity_score'])

        # 读取并显示标注信息
        
        
        
    else:
        st.error(f"无法找到视频文件: {selected_video}")

# 主函数
def main():
    st.title("视频注释对比展示")
    
    # 加载分数数据和标注数据
    score_data = load_score_data(score_csv)
    caption_data = load_caption_data(caption_csv)
    
    # 下拉选择视频名称
    video_names = score_data['video_name'].unique()
    selected_video = st.selectbox("选择一个视频", video_names)
    
    if selected_video:
        display_video_info(score_data, caption_data, selected_video)

if __name__ == "__main__":
    main()
