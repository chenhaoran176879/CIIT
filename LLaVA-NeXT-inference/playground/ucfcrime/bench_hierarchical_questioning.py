anomaly_detection_question="""
Is there any abnormal event that might be related to crime or danger in this video?
You should answer start with Yes or No. Then start a new line and state the reason why.
Your answer:
"""
# # 不需要gpt评分

# integrated_anomaly_detection_question="""
# Read this video and decide if there is any abnormal event that might be related to crime or danger.
# If everything is normal, please respond with Normal.
# Else if you find a dangerous event occurred in this video.

# """

crime_classification_question="""
We believe that a dangerous event occurred in this video.
And this event might belong to one or more categories of the crime below: 
Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vadalism.
Your Answer should only consist of the categories above with a form like ['RoadAccidents'] and ["Abuse","Arrest",...].
For example, if there is a car driving on the road and it explodes from inside, your answer should be ["RoadAccidents","Explosion"]

Now analyze this video carefully and classify it based on the events and human activities. 

Your answer: 
"""
# 这样设置的目的是绕开模型可能存在的对危险概念的盲区，这个问题只为非normal案例而设置。
# 不需要gpt评分


event_description_question="""
Describe the events in this video. 
You are expected to first detect whether this video contains abnormal events or only normal events, \
and then give a short description of the detected events. 
Last, you should provide enough details about the events, especially environment and human activity.

Your Answer:
"""

eval_event_description_prompt="""
Compare a ground truth from human and an answer from AI models, and give a score for models' answer. 
Both are description texts of the same video. 
The score for each answer ranges from 0 to 100, based on how well the AI's description matches the ground truth.
Consider the following criteria when scoring:
1.  Does the AI-generated description correctly indicate whether the video contains any abnormal or only normal events? 
    -- Maximum Score: 30 points
    -- If the ground truth only contains abnormal events and the AI detects any abnormal events, award the full 30 points.
    -- Else if the ground truth only contains normal events and the AI correctly identifies this, award the full 30 points.
    
2.  Does the AI-generated description correctly provide the key details of the events, including environment, human appearance, actions, objects?
    -- Maximum Score: 70 points
    -- Up to 40 points if the AI indicates the factual content of the ground truth. 
    -- Up to 20 points if the AI indicates further reasonable details that are not included in the ground truth but are logically consistent with the video content.
    -- Up to 10 points if the AI is clear, well-organized, and free of ambiguities or confusing statements.

Description 1 (ground truth): {}

Description 2 (AI): {}

Now give your score, reponse with only a number:
"""
# and reasons in form of {{"score":99, "reasons":[reason1,reason2,...]}}. The reasons must be in Chinese and is preferred te contain details.


event_description_with_classification="""
We believe that an event of {} happens in this video.
You are expected to firstly give a short description of this abnormal event. \
Then, you should provide enough details about the events, especially environment and human activity.

Your Answer:
"""

eval_event_description_with_classification_prompt="""
Compare a ground truth from human and an answer from AI models, and give a score for models' answer. 
Both are description texts of the same video. 
The score for each answer ranges from 0 to 100, based on how well the AI's description matches the ground truth.
Consider the following criteria when scoring:
1.  Does the AI-generated description correctly indicate whether the video contains any abnormal or only normal events? 
    -- Maximum Score: 30 points
    -- If the ground truth only contains abnormal events and the AI detects any abnormal events, award the full 30 points.
    -- Else if the ground truth only contains normal events and the AI correctly identifies this, award the full 30 points.
    
2.  Does the AI-generated description correctly provide the key details of the events, including environment, human appearance, actions, objects?
    -- Maximum Score: 70 points
    -- Up to 40 points if the AI indicates the factual content of the ground truth. 
    -- Up to 20 points if the AI indicates further reasonable details that are not included in the ground truth but are logically consistent with the video content.
    -- Up to 10 points if the AI is clear, well-organized, and free of ambiguities or confusing statements.

Description 1 (ground truth): {}

Description 2 (AI): {}

Now give your score, reponse with only a number:
"""



temporal_grounding_question="""
We believe that a dangerous event occurred in this video. And it is recognized by human to as {}.
Please find out when this event starts and ends. Provide the start and end times in seconds in your answer in a format of {{"start_time":4, "end_time":15}}.
Do not give other response or format.

Your answer:
"""

# 按照目前的视频采样方法，模型是没有秒的概念的，所以这个问题应该暂时屏蔽。后续为了增加论文工作，可以考虑告诉视频模型采样方法，从而让它计算每一帧对应的时间。
