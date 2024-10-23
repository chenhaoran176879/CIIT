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

For an abnormal example, your answer could be like: Severe physical abuse and domination are depicted in the video. A woman in floral clothing forcefully restrains and assaults a man in a blue top, using a pillow to smother his face, preventing him from breathing. She continues to dominate him, forcefully feeding him with a spoon, slapping his face, and controlling his movements.
For a normal exmaple, your answer could be like: No dangerous event detected in the video. The footage captures a scene on a multi-lane highway. Initially, the left lanes have fewer vehicles while the right lane experiences congestion. As time progresses, various colored cars, including blue-green and white vans, navigate the lanes, with some performing lane changes.
Your Answer:
"""

eval_event_description_prompt="""
Compare a ground truth from human and an answer from AI models, and give a score for models' answer. 
Both are description texts of the same video. 
The score for each answer ranges from 0 to 100, based on how well the AI's description matches the ground truth.
Use the following 2 criteria and add up to a final score:
1. This task is focused on detecting possible dangerous events in the video.
First, analyze the ground truth to determine whether there is any dangerous event, possible violence, or crime in the video.
Then, analyze the AI's description based on the same criteria.
    --30 points: If the AI agrees with the ground truth regarding the presence or absence of danger, add 30 points.Otherwise, add 0 points.


2.  Does the AI-generated description correctly provide the key details of the events, including environment, human appearance, actions, and objects?
    Maximum Score: 70 points
    --30 points: Based on how many the AI objects, environment, or movements mentions that are also mentioned in the ground truth, like car, animal, human. 
    --20 points: Based on AI's additional reasonable details not present in the ground truth but logically consistent with the video content.
    --20 points: Based on how AI describe the story. Closer to ground truth, higher the points.

Add these 4 parts of points to your final score.


Example 1:
Description 1 (ground truth):A law enforcement operation is conducted in a room. A man in black, wearing purple gloves, leads a handcuffed man in a black vest and shorts into the space and pushes him against the wall. The man in black searches the suspect multiple times, placing items on a table. A bald man in black arrives and organizes the items. The man in the vest removes his shoes and socks, which are taken away by others. Later, the handcuffs are removed, and the man in the vest is assisted by two individuals. Throughout the process, a police officer briefly interacts with someone at a window before leaving. The man in black and the man in the vest exit together.
Description 2 (AI):The video depicts a scene inside a police station. A man is seen standing against a wall, and two officers approach him. One of the officers then proceeds to handcuff the man, and the other officer assists by holding the man's arms. The man is then led away by the officers. The environment is a police station with white walls and a desk in the background. The officers are wearing black uniforms, and the man is wearing a black tank top and shorts.
Your score: 75

Example 2:
Description 1 (ground truth): A burger occurred in a small store. A masked individual shattered the glass door to gain entry. Subsequently, three to four people appeared in the scene, holding shopping bags filled with items. They searched the store and quickly fled.
Description 2 (AI):The video shows a normal event. The scene depicts a woman walking down a set of stairs and entering a store. She then walks around the store and exits through the same door.
Your score: 20

Example 3:
Description 1 (ground truth): An incident of vandalism occurred in a store. A naked man entered and approached the counter, scanning the area. He abruptly threw an item onto the floor, then proceeded to topple shelves as he moved further into the store. A man in white briefly appeared, possibly reacting to the situation, and exited. Meanwhile, a person emerged from behind the counter. Two additional individuals, one in black and another in a striped shirt with a hat, left the scene. The vandal eventually left the store after causing more destruction.
Description 2 (AI):The video depicts a robbery in a convenience store. A man enters the store and approaches the counter. He then proceeds to smash the cash register with a hammer, causing it to fall onto the floor. The man then runs out of the store, leaving behind a trail of destruction. The video captures the chaotic scene as the man's actions cause damage to the store's interior.
Your score: 85 




Question:
Description 1 (ground truth): {}
Description 2 (AI): {}

Your score:
"""
# and reasons in form of {{"score":99, "reasons":[reason1,reason2,...]}}. The reasons must be in Chinese and is preferred te contain details.


event_description_with_classification="""
We believe that an event of {} happens in this video.
You are expected to firstly give a description of this kind of event. Then, you should provide enough details about the events.
For an abnormal example, your answer could be like: Severe physical abuse and domination are depicted in the video. A woman in floral clothing forcefully restrains and assaults a man in a blue top, using a pillow to smother his face, preventing him from breathing. She continues to dominate him, forcefully feeding him with a spoon, slapping his face, and controlling his movements.
For a normal exmaple, your answer could be like: No dangerous event detected in the video. The footage captures a scene on a multi-lane highway. Initially, the left lanes have fewer vehicles while the right lane experiences congestion. As time progresses, various colored cars, including blue-green and white vans, navigate the lanes, with some performing lane changes.
Your Answer to this video:
"""

eval_event_description_with_classification_prompt="""
Compare a ground truth from human and an answer from AI models, and give a score for models' answer. 
Both are description texts of the same video. 
The score for each answer ranges from 0 to 100, based on how well the AI's description matches the ground truth.
Use the following 2 criteria and add up to a final score:
1. This task is focused on detecting possible dangerous events in the video.
First, analyze the ground truth to determine whether there is any dangerous event, possible violence, or crime in the video.
Then, analyze the AI's description based on the same criteria.
    --30 points: If the AI agrees with the ground truth regarding the presence or absence of danger, add 30 points.Otherwise, add 0 points.


2.  Does the AI-generated description correctly provide the key details of the events, including environment, human appearance, actions, and objects?
    Maximum Score: 70 points
    --30 points: Based on how many the AI objects, environment, or movements mentions that are also mentioned in the ground truth, like car, animal, human. 
    --20 points: Based on AI's additional reasonable details not present in the ground truth but logically consistent with the video content.
    --20 points: Based on how AI describe the story. Closer to ground truth, higher the points.

Add these 4 parts of points to your final score.


Example 1:
Description 1 (ground truth):A law enforcement operation is conducted in a room. A man in black, wearing purple gloves, leads a handcuffed man in a black vest and shorts into the space and pushes him against the wall. The man in black searches the suspect multiple times, placing items on a table. A bald man in black arrives and organizes the items. The man in the vest removes his shoes and socks, which are taken away by others. Later, the handcuffs are removed, and the man in the vest is assisted by two individuals. Throughout the process, a police officer briefly interacts with someone at a window before leaving. The man in black and the man in the vest exit together.
Description 2 (AI):The video depicts a scene inside a police station. A man is seen standing against a wall, and two officers approach him. One of the officers then proceeds to handcuff the man, and the other officer assists by holding the man's arms. The man is then led away by the officers. The environment is a police station with white walls and a desk in the background. The officers are wearing black uniforms, and the man is wearing a black tank top and shorts.
Your score: 75

Example 2:
Description 1 (ground truth): A burger occurred in a small store. A masked individual shattered the glass door to gain entry. Subsequently, three to four people appeared in the scene, holding shopping bags filled with items. They searched the store and quickly fled.
Description 2 (AI):The video shows a normal event. The scene depicts a woman walking down a set of stairs and entering a store. She then walks around the store and exits through the same door.
Your score: 20

Example 3:
Description 1 (ground truth): An incident of vandalism occurred in a store. A naked man entered and approached the counter, scanning the area. He abruptly threw an item onto the floor, then proceeded to topple shelves as he moved further into the store. A man in white briefly appeared, possibly reacting to the situation, and exited. Meanwhile, a person emerged from behind the counter. Two additional individuals, one in black and another in a striped shirt with a hat, left the scene. The vandal eventually left the store after causing more destruction.
Description 2 (AI):The video depicts a robbery in a convenience store. A man enters the store and approaches the counter. He then proceeds to smash the cash register with a hammer, causing it to fall onto the floor. The man then runs out of the store, leaving behind a trail of destruction. The video captures the chaotic scene as the man's actions cause damage to the store's interior.
Your score: 85 

Example 4:



Question:
Description 1 (ground truth): {}
Description 2 (AI): {}

Your score:
"""



temporal_grounding_question="""
We believe that a dangerous event occurred in this video. And it is recognized by human to as {}.
Please find out when this event starts and ends. Provide the start and end times in seconds in your answer in a format of {{"start_time":4, "end_time":15}}.
Do not give other response or format.

Your answer:
"""

# 按照目前的视频采样方法，模型是没有秒的概念的，所以这个问题应该暂时屏蔽。后续为了增加论文工作，可以考虑告诉视频模型采样方法，从而让它计算每一帧对应的时间。
