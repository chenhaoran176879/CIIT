import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore", message="Please replace `st.experimental_get_query_params` with `st.query_params`.")


def write_jsonl(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:  # 使用'a'模式来追加数据
        for obj in data:
            json_line = json.dumps(obj) + '\n'  # 将对象转换为JSON字符串，并添加换行符
            file.write(json_line)

# 设置保存数据的文件路径
BASE_DIR = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/"  # 服务器上视频文件的路径
 # json 备份
DATA_BASE = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/human_annatation_train.jsonl"

crime_dict = {
    "Abuse": "虐待",
    "Explosion": "爆炸",
    "Stealing": "偷窃",
    "Fighting": "打架",
    "Arrest": "逮捕",
    "RoadAccidents": "交通事故",
    "Vandalism": "故意破坏",
    "Arson": "纵火",
    "Robbery": "抢劫",
    "Assault": "袭击",
    "Shooting": "枪击",
    "Burglary": "入室盗窃",
    "Shoplifting": "商店行窃"
}


# 界面标题
st.title("UCF-Crime犯罪视频标注工具")
st.write("联系人：陈浩然@武智院企业微信")
st.write("请结合文档https://kdocs.cn/l/ck0tWISsoIhq进行标注")

page = st.selectbox("选择页面", ["新视频标注", "查看和修改已标注数据"])
# 列出主文件夹中的所有子文件夹
if page == "新视频标注":
    folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    folders.sort()
    # 选择子文件夹
    selected_folder = st.selectbox("选择一个犯罪/危险视频分类", folders)

    if selected_folder:
        folder_path = os.path.join(BASE_DIR, selected_folder)
        
        # 列出子文件夹中的所有视频文件
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        video_files_no_annot = []
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            save_path = video_path.split('.')[0]+'_annotation.csv'
            save_path_temp = video_path.split('.')[0]+'_annotation_temp.csv'
            if not os.path.exists(save_path):
                video_files_no_annot.append(video_file)
            else:
                file_creation_time = datetime.fromtimestamp(os.path.getctime(save_path))
                time_threshold = datetime.now() - timedelta(minutes=10)
                df = pd.read_csv(save_path)
                if df.empty:
                    video_files_no_annot.append(video_file)

        video_files_no_annot.sort()
        # 选择要标注的视频
        selected_video = st.selectbox("选择一个视频文件进行标注", video_files_no_annot)

        if selected_video:
            video_path = os.path.join(folder_path, selected_video)
            
            save_path = video_path.split('.')[0]+'_annotation.csv'
            if not os.path.exists(save_path):
                df = pd.DataFrame(columns=["video_name", "start_time", "end_time", "description", "quality_score", "confidence_score", "timestamp", "is_repeated", "first_end_time"])
                df.to_csv(save_path, index=False)

            # 显示选中的视频
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            col1, col2 = st.columns(2)
            with col1:
                # 时间戳记录
                st.write("请标记事件起始和结束时间")
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
                description = st.text_area(f"描述这个视频片段中的{selected_folder}（{crime_dict[selected_folder]}）犯罪/危险行为", height=350)

                # 视频重复性检查
                is_repeated = st.radio("这个视频是否多次重复播放？", ["否", "是"])
                if is_repeated == "是":
                    st.write('请标记你选取的视频片段的开始时间')
                    first_start_col1, first_start_col2 = st.columns([1, 3])
                    with first_start_col1:
                        first_start_minutes = st.number_input("分钟", min_value=0, format="%d", key="first_start_min")
                    with first_start_col2:
                        first_start_seconds = st.number_input("：秒钟", min_value=0, max_value=59, format="%02d", key="first_start_sec")
                    first_start_time = first_start_minutes * 60 + first_start_seconds
                    st.write('请标记你选取的视频片段的结束时间')
                    first_end_col1, first_end_col2 = st.columns([1, 3])
                    with first_end_col1:
                        first_end_minutes = st.number_input("分钟", min_value=0, format="%d", key="first_end_min")
                    with first_end_col2:
                        first_end_seconds = st.number_input("：秒钟", min_value=0, max_value=59, format="%02d", key="first_end_sec")
                    first_end_time = first_end_minutes * 60 + first_end_seconds
                else:
                    first_start_time = None
                    first_end_time = None

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
                    "timestamp": timestamp,
                    "is_repeated": is_repeated,
                    "first_start_time": first_start_time,
                    "first_end_time": first_end_time
                }

                new_row = pd.DataFrame([new_data])

                # 使用 concat 方法将新数据行添加到 DataFrame 中
                df = pd.concat([df, new_row], ignore_index=True)

                # 保存到CSV
                df.to_csv(save_path, index=False)

               

                with open(DATA_BASE,'a',encoding = 'utf-8') as f:
                    f.write(json.dumps(new_data)+'\n')
                        

                # 显示成功提示
                success_message = st.empty()
                success_message.success("标注已保存")

                # 等待一秒钟然后清除成功提示
                time.sleep(1)
                success_message.empty()
                st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
                # 模拟刷新页面

elif page == '查看和修改已标注数据':
    folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    folders.sort()
    # 选择子文件夹
    selected_folder = st.selectbox("选择一个犯罪/危险视频分类", folders)
    if selected_folder:
        folder_path = os.path.join(BASE_DIR, selected_folder)
        annotation_files = [f for f in os.listdir(folder_path) if f.endswith('_annotation.csv')]
        annotation_files.sort()
        annotation_files_not_empty = []
        for annotation_file in annotation_files:
            annotation_path = os.path.join(folder_path, annotation_file)
            df = pd.read_csv(annotation_path)
            if not df.empty:
                annotation_files_not_empty.append(annotation_file)
        annotation_files_not_empty.sort()
        selected_annotation_file = st.selectbox("选择一个标注文件", annotation_files_not_empty)
        if selected_annotation_file:
            print(selected_annotation_file)
            annotation_path = os.path.join(folder_path, selected_annotation_file)
            df = pd.read_csv(annotation_path)
            st.write("当前标注数据:")
            table_place_holder = st.empty()
            table_place_holder.dataframe(df)

            video_prefix = selected_annotation_file.split('_annotation.csv')[0] 
            print(video_prefix)
            video_path = None
            for f in os.listdir(folder_path):
                    
                if f.startswith(video_prefix) and f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(folder_path,f)
                    print(video_path)
                    break
            

            try:
                
                st.video(f"https://zidongtaichu.obs.cn-central-221.ovaijisuan.com/picasso/ucf_crime_train/{selected_folder}/{video_path.split('/')[-1]}")
                st.write(f"找到视频网址: https://zidongtaichu.obs.cn-central-221.ovaijisuan.com/picasso/ucf_crime_train/{selected_folder}/{video_path.split('/')[-1]}")
            except Exception  as e:
                if video_path:
                    st.write(f"找到视频文件: {video_path}")
                else:
                    st.write("未找到匹配的视频文件")

                with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)

        

        
            record_index = st.selectbox("选择要修改的标注记录序号", df.index)#index=0)
            if record_index is not None:
            # 显示选中的标注记录
                selected_record = df.loc[record_index]

                if st.button("删除这条记录"):
                    # 删除选定的记录
                    df = df.drop(record_index)

                    # 重新写入CSV文件
                    df.to_csv(annotation_path, index=False, encoding='utf-8')

                    st.success("标注记录已删除")
                    st.dataframe(df)
                    table_place_holder.dataframe(df)
                    time.sleep(1)
                    #st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)


            start_time = st.number_input("起始时间 (秒)", value=selected_record["start_time"])
            end_time = st.number_input("结束时间 (秒)", value=selected_record["end_time"])
            quality_score = st.selectbox("质量评分", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=int(selected_record["quality_score"]) - 1)
            confidence_score = st.selectbox("置信度评分", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=int(selected_record["confidence_score"]) - 1)
            description = st.text_area("描述", value=selected_record["description"],height=300)
            is_repeated = st.radio("这个视频是否多次重复播放？", ["否", "是"], index=0 if selected_record["is_repeated"] == "否" else 1)
            if is_repeated == "是":
                st.write('请标记你选取的视频片段的开始时间')
                first_start_col1, first_start_col2 = st.columns([1, 3])
                with first_start_col1:
                    first_start_minutes = st.number_input("分钟", min_value=0, format="%d", key="first_start_min",value=int(selected_record.get("first_start_time", 0)//60))
                with first_start_col2:
                    first_start_seconds = st.number_input("：秒钟", min_value=0, max_value=59, format="%02d", key="first_start_sec",value=int(selected_record.get("first_start_time", 0)%60))
                first_start_time = first_start_minutes * 60 + first_start_seconds
                st.write('请标记你选取的视频片段的结束时间')
                first_end_col1, first_end_col2 = st.columns([1, 3])
                with first_end_col1:
                    first_end_minutes = st.number_input("分钟", min_value=0, format="%d", key="first_end_min",value=int(selected_record.get("first_end_time", 0)//60))
                with first_end_col2:
                    first_end_seconds = st.number_input("：秒钟", min_value=0, max_value=59, format="%02d", key="first_end_sec",value=int(selected_record.get("first_end_time", 0)%60))
                first_end_time = first_end_minutes * 60 + first_end_seconds
            else:
                first_start_time = None
                first_end_time = None

            if st.button("保存修改"):
                # 获取当前时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 创建一个包含修改后的数据的新记录
                modified_record = df.loc[record_index].copy()
                
                new_data = {
                        "video_name": modified_record['video_name'],
                        "start_time": start_time,
                        "end_time": end_time,
                        "description": description,
                        "quality_score": quality_score,
                        "confidence_score": confidence_score,
                        "timestamp": timestamp,
                        "is_repeated": is_repeated,
                        "first_start_time": first_start_time,
                        "first_end_time": first_end_time
                    }
                
                
                # 将修改后的记录添加到 DataFrame 的末尾
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df,new_row], ignore_index=True)
                
                # 重新写入整个 DataFrame 到 CSV 文件中，使用 'w' 模式覆盖原文件
                df.to_csv(annotation_path, mode='w', index=False, encoding='utf-8')
                with open(DATA_BASE,'a',encoding = 'utf-8') as f:
                        f.write(json.dumps(new_data)+'\n')
                st.success("标注记录已更新，并保存到文件末尾")
                table_place_holder.dataframe(df)
                st.dataframe(df)
                time.sleep(1)
                #st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)





