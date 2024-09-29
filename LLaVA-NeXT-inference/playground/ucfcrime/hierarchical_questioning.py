anomaly_detection_question="""
Is there any abnormal event that might be related to crime or danger in this video?
You should answer only Yes or No.
"""


action_description_question="""
Describe the action(s) of the main character(s) in this video.
"""


crime_classification_question="""
We believe that a dangerous event occurred in this video.
And it belongs to one or more categories of the crime below: 
Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing and Vadalism. Normal(If no dangerous event).
Watch this video carefully and your Answer should only consist of the categories above.

Answer: 
"""


temporal_grounding_question="""
We believe that a dangerous event occurred in this video. And it is recognized by human as {}.
Please find out when this event starts and ends. Provide the start and end times in seconds in your answer in a format of {"start_time":4, "end_time":15}.
Do not give other response or format.
Your answer:
"""





