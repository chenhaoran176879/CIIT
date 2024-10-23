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
You are expected to detect whether this video contains abnormal events or only normal events, \
and then give a description of the detected events with details, especially environmen human looking and action.
Below are two examples of your answer.
Your answer: Severe physical abuse and domination are depicted in the video. A woman in floral clothing forcefully restrains and assaults a man in a blue top, using a pillow to smother his face, preventing him from breathing. She continues to dominate him, forcefully feeding him with a spoon, slapping his face, and controlling his movements.
Your answer: No dangerous event detected in the video. The footage captures a scene on a multi-lane highway. Initially, the left lanes have fewer vehicles while the right lane experiences congestion. As time progresses, various colored cars, including blue-green and white vans, navigate the lanes, with some performing lane changes.

Your should not use the same expression in example answers. You should only learn how to organize your answer and how specific your details should be. 
Now watch the video I give you, and write your answer.
Your Answer:"""



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
    --30 points: Based on how many objects, environment, human or movements that are mentioned in the ground truth are captured by AI.
    --20 points: Based on AI's additional reasonable details not present in the ground truth but logically consistent with the video content.
    --20 points: Based on how AI describe the story. Closer to ground truth, higher the points.

Add these parts of points to your final score.

Example 1:
Description 1 (ground truth): A burger occurred in a small store. A masked individual shattered the glass door to gain entry. Subsequently, three to four people appeared in the scene, holding shopping bags filled with items. They searched the store and quickly fled.
Description 2 (AI):The video shows a normal event. The scene depicts a woman walking down a set of stairs and entering a store. She then walks around the store and exits through the same door.
Your score: 20


Example 2:
Description 1 (ground truth):An armed robbery occurred in the scene. Initially, three men were present, with two of them brandishing guns and engaging in a tense standoff behind a glass counter, while the third man watched from the doorway. One of the gunmen eventually rushed out, forcing the other to retreat quickly. The third man cautiously inspected the area before rushing back inside. Moments later, another individual briefly appeared and retreated, followed by one of the gunmen returning to the scene. The final scene shows a man exiting, bending down to pick up an item, and going back into the house. They did not open fire.
Description 2 (AI):The video depicts a severe physical altercation and robbery in a convenience store. A man in a black jacket and white cap enters the store and confronts the store clerk. The man in the black jacket points a gun at the clerk, demanding money. The clerk, visibly frightened, complies with the demand. The assailant then proceeds to hit the clerk with the gun, causing the clerk to fall to the ground. The assailant continues to hit the clerk multiple times, displaying a high level of aggression and violence.\n\nThroughout the video, the assailant maintains control over the situation, using the gun to intimidate and physically assault the clerk. The store's shelves and products are visible in the background, indicating the setting is a small convenience store. 
﻿
Your score: 60

Example 3:
Description 1 (ground truth): An incident of vandalism occurred in a store. A naked man entered and approached the counter, scanning the area. He abruptly threw an item onto the floor, then proceeded to topple shelves as he moved further into the store. A man in white briefly appeared, possibly reacting to the situation, and exited. Meanwhile, a person emerged from behind the counter. Two additional individuals, one in black and another in a striped shirt with a hat, left the scene. The vandal eventually left the store after causing more destruction.
Description 2 (AI):The video depicts a robbery in a convenience store. A man enters the store and approaches the counter. He then proceeds to smash the cash register with a hammer, causing it to fall onto the floor. The man then runs out of the store, leaving behind a trail of destruction. The video captures the chaotic scene as the man's actions cause damage to the store's interior.
Your score: 80

Example 4:
Description 1 (ground truth):A road accident occurred where a red car collided with a motorcycle, resulting in the car driver exiting to inspect the damage, and nearby pedestrians also approaching to investigate the scene.
Description 2 (AI):Abnormal event detected.\n\nThe video captures a scene on a city street where a motorcyclist is involved in a collision with a red car. The motorcyclist is thrown off the bike and lands on the ground, appearing to be injured. The red car comes to a stop near the injured motorcyclist. Several pedestrians, including a man in a pink jacket and a woman in a pink coat, are seen walking by the scene. A cyclist also passes by the accident site. The environment is urban with buildings and parked cars in the background. The incident appears to have occurred in the middle of the street, causing a disruption in the flow of traffic.
Your score: 100



Question:
Description 1 (ground truth): {}
Description 2 (AI): {}

Your score: """


# and reasons in form of {{"score":99, "reasons":[reason1,reason2,...]}}. The reasons must be in Chinese and is preferred te contain details.


event_description_with_classification="""
We believe that an event of {} happens in this video.
You are expected to firstly give a description of this kind of event. 
Then give a description of the detected events with details, especially environmen human looking and action.
Below are two examples of your answer.
Example answer 1: Severe physical abuse and domination are depicted in the video. A woman in floral clothing forcefully restrains and assaults a man in a blue top, using a pillow to smother his face, preventing him from breathing. She continues to dominate him, forcefully feeding him with a spoon, slapping his face, and controlling his movements.
Example answer 2: No dangerous event detected in the video. The footage captures a scene on a multi-lane highway. Initially, the left lanes have fewer vehicles while the right lane experiences congestion. As time progresses, various colored cars, including blue-green and white vans, navigate the lanes, with some performing lane changes.

Your should not use the same expression in example answers. You should only learn how to organize your answer and how specific your details should be. 
Now watch the video I give you, and write your answer.
Your Answer:"""

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
    --30 points: Based on how many objects, environment, human or movements that are mentioned in the ground truth are captured by AI.
    --20 points: Based on AI's additional reasonable details not present in the ground truth but logically consistent with the video content.
    --20 points: Based on how AI describe the story. Closer to ground truth, higher the points.

Add these parts of points to your final score.

Example 1:
Description 1 (ground truth): A burger occurred in a small store. A masked individual shattered the glass door to gain entry. Subsequently, three to four people appeared in the scene, holding shopping bags filled with items. They searched the store and quickly fled.
Description 2 (AI):The video shows a normal event. The scene depicts a woman walking down a set of stairs and entering a store. She then walks around the store and exits through the same door.
Your score: 20


Example 2:
Description 1 (ground truth):An armed robbery occurred in the scene. Initially, three men were present, with two of them brandishing guns and engaging in a tense standoff behind a glass counter, while the third man watched from the doorway. One of the gunmen eventually rushed out, forcing the other to retreat quickly. The third man cautiously inspected the area before rushing back inside. Moments later, another individual briefly appeared and retreated, followed by one of the gunmen returning to the scene. The final scene shows a man exiting, bending down to pick up an item, and going back into the house. They did not open fire.
Description 2 (AI):The video depicts a severe physical altercation and robbery in a convenience store. A man in a black jacket and white cap enters the store and confronts the store clerk. The man in the black jacket points a gun at the clerk, demanding money. The clerk, visibly frightened, complies with the demand. The assailant then proceeds to hit the clerk with the gun, causing the clerk to fall to the ground. The assailant continues to hit the clerk multiple times, displaying a high level of aggression and violence.\n\nThroughout the video, the assailant maintains control over the situation, using the gun to intimidate and physically assault the clerk. The store's shelves and products are visible in the background, indicating the setting is a small convenience store. 
﻿
Your score: 60

Example 3:
Description 1 (ground truth): An incident of vandalism occurred in a store. A naked man entered and approached the counter, scanning the area. He abruptly threw an item onto the floor, then proceeded to topple shelves as he moved further into the store. A man in white briefly appeared, possibly reacting to the situation, and exited. Meanwhile, a person emerged from behind the counter. Two additional individuals, one in black and another in a striped shirt with a hat, left the scene. The vandal eventually left the store after causing more destruction.
Description 2 (AI):The video depicts a robbery in a convenience store. A man enters the store and approaches the counter. He then proceeds to smash the cash register with a hammer, causing it to fall onto the floor. The man then runs out of the store, leaving behind a trail of destruction. The video captures the chaotic scene as the man's actions cause damage to the store's interior.
Your score: 80

Example 4:
Description 1 (ground truth):A road accident occurred where a red car collided with a motorcycle, resulting in the car driver exiting to inspect the damage, and nearby pedestrians also approaching to investigate the scene.
Description 2 (AI):Abnormal event detected.\n\nThe video captures a scene on a city street where a motorcyclist is involved in a collision with a red car. The motorcyclist is thrown off the bike and lands on the ground, appearing to be injured. The red car comes to a stop near the injured motorcyclist. Several pedestrians, including a man in a pink jacket and a woman in a pink coat, are seen walking by the scene. A cyclist also passes by the accident site. The environment is urban with buildings and parked cars in the background. The incident appears to have occurred in the middle of the street, causing a disruption in the flow of traffic.
Your score: 100



Question:
Description 1 (ground truth): {}
Description 2 (AI): {}

Your score: """



temporal_grounding_question="""
We believe that a dangerous event occurred in this video. And it is recognized by human to as {}.
Please find out when this event starts and ends. Provide the start and end times in seconds in your answer in a format of {{"start_time":4, "end_time":15}}.
Do not give other response or format.

Your answer:
"""

# 按照目前的视频采样方法，模型是没有秒的概念的，所以这个问题应该暂时屏蔽。后续为了增加论文工作，可以考虑告诉视频模型采样方法，从而让它计算每一帧对应的时间。


multiple_choice_question="""
Please watch this video carefully and answer a multiple choice question.
The question is composed of a stem and 4 options. Only 1 option is true.
Choose the best option based on the story and details in this video. 
You do not need to explain why. You should answer only one letter, A,B,C or D.
Below is the question.
{}
Your answer:"""

# multiple_choice_question="""
# Please watch this video carefully and answer the following 5 multiple choice questions.
# Each question is composed of a stem and 4 options. Only 1 option is true.
# Choose the best option based on the story and details in this video. 
# After the 5 questions, you should give your answer in a form of ['A','C','D','B','C']. 
# These five letters denote your choice for the 5 quetsions in the corresponding sequence.
# Do not involve other sentence or words. Just answer with this list form.


# Below are the 5 multiple choice questions:
# {}

# Your answer:
# """