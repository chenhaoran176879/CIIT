import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# 设置保存数据的文件路径

BASE_DIR = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/"  # 服务器上视频文件的路径



# 初始化CSV文件，如果不存在的话


# 界面标题
st.title("视频标注工具")
st.write("联系人：陈浩然@武智院企业微信")

# 列出主文件夹中的所有子文件夹
folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
folders.sort()
# 选择子文件夹
selected_folder = st.selectbox("选择一个犯罪视频分类", folders)

if selected_folder:
    folder_path = os.path.join(BASE_DIR, selected_folder)
    
    # 列出子文件夹中的所有视频文件
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files_no_annot = []
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        save_path = video_path.split('.')[0]+'_annotation.csv'
        if not os.path.exists(save_path):
            video_files_no_annot.append(video_file)
        else:
            file_creation_time = datetime.fromtimestamp(os.path.getctime(save_path))
            time_threshold = datetime.now() - timedelta(minutes=10)
            df = pd.read_csv(save_path)
            if df.empty and file_creation_time < time_threshold:
                video_files_no_annot.append(video_file)

    video_files_no_annot.sort()
    # 选择要标注的视频
    selected_video = st.selectbox("选择一个视频文件进行标注", video_files_no_annot)


    if selected_video:
        video_path = os.path.join(folder_path, selected_video)
        
        save_path = video_path.split('.')[0]+'_annotation.csv'
        if not os.path.exists(save_path):
            df = pd.DataFrame(columns=["video_name", "start_time", "end_time", "description", "quality_score", "confidence_score", "timestamp"])
            df.to_csv(save_path, index=False)

        # 显示选中的视频
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
        

        col1, col2 = st.columns(2)
        with col1:
            # 时间戳记录
            st.write("请在视频中标记起始和结束时间")
            start_col1, start_col2 = st.columns([1, 3])
            with start_col1:
                start_minutes = st.number_input("起始分钟", min_value=0, format="%d", key="start_min")
            with start_col2:
                start_seconds = st.number_input("：起始秒钟", min_value=0, max_value=59, format="%02d", key="start_sec")

            # 终止时间
            end_col1, end_col2 = st.columns([1, 3])
            with end_col1:
                end_minutes = st.number_input("结束分钟", min_value=0, format="%d", key="end_min")
            with end_col2:
                end_seconds = st.number_input("：结束秒钟", min_value=0, max_value=59, format="%02d", key="end_sec")

            start_time = start_minutes * 60 + start_seconds
            end_time = end_minutes * 60 + end_seconds

            # 打分
            quality_score = st.selectbox("质量评分", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            # 置信度评分（使用星级评分）
            confidence_score = st.selectbox("置信度评分", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            # 描述
        with col2:
            description = st.text_area("描述这个视频片段",height=350)



        # 保存按钮
        if st.button("保存标注"):
            # 获取当前时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 加载现有数据
            df = pd.read_csv(save_path)

            # 添加新数据
            new_data = {
                "video_name": selected_video,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "quality_score": quality_score,
                "confidence_score": confidence_score,
                "timestamp": timestamp
            }

            df = df.append(new_data, ignore_index=True)

            # 保存到CSV
            df.to_csv(save_path, index=False)

            st.success("标注已保存！")
