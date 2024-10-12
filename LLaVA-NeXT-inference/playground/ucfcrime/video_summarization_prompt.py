video_summarization_prompt = """
I need you to summarize a few descriptions about an abnormal video. The video's name is given first, \
which means the classification of the abnormal event inside and the index of the video. You need to pay attention to this classification. \
Every description begins with 2 timestamps marking the start and end time of this description.\
All descriptions add up to a complete video. 
I will give you 1 examples from human annotators and then you straightly give your answer to the last video with the same format.
The answer should include video name, start time and end time which means the time count by second when the abnormal starts and ends, \
(Please give your best guess) and a highly summarized description of the abnormal event. At the beginning of your description, you should \
define the classification of the event, like Abuse, Explosion or Stealing. You should also be more specific and comprehensive about this classification, for example, \
"Violent robbery, abuse, and assault were committed in the video". Afterwards, you describe the event.


Example 1:
('Abuse001_x264', ['00:00.0 00:05.3 ##A woman with short hair, slightly fat, wearing a white top and black pants stood in front of the table, \
picked up a book from the table, and opened it to read', '00:07.0 00:08.5 ##A man wearing a white shirt and black pants entered the house and \
walked towards the short-haired and fat woman in front who was reading a book.', '00:07.2 00:08.5 ##A man wearing a black shirt and black pants\
entered the house and walked towards the short-haired and fat woman in front who was reading a book.', \
"00:08.2 00:08.9 ##A man wearing a white shirt and black pants approached a short-haired, \
fat woman wearing a white shirt and black pants. When the woman was not paying attention, \
the man suddenly pulled out a piece of red cloth from the left side of the woman's body, then turned and ran away.",\
'00:08.9 00:11.2 ##A man in black clothes approached a short-haired, fat woman wearing a white top and black pants.\
When the woman turned back to the right, he punched the woman in the head and ran out. .',\
'00:08.9 00:11.2 ##The woman fell to the ground in pain, and the book in her hand fell to the ground when she fell.\
At the same time, she knocked the red wooden table in front of her crookedly. There were three things on the table.', \
'00:11.3 00:13.3 ##A woman with short hair and a fat figure wearing a white top and black pants fell to the ground,\
raised her right hand to touch her forehead, and hit her head on the wooden table leg.',\
'00:15.2 00:18.9 ##A woman with short hair, slightly fat, wearing a white top and black pants fell to the ground.\
She touched her forehead with her right hand and looked towards the direction of entering the house.\
She retracted her left leg, supported her left hand on the ground, and held her forehead with her right hand. Upper body.', \
'00:19.7 00:25.4 ##A woman with short hair, slightly fat, wearing a white top and black pants is sitting on the ground, \
leaning on the ground with her left hand, her right hand dropped from her head and placed on the knee of her right leg,\
her right leg is bent, she turns her head and kicks with her left leg Kneel down on the ground, stand up with your right hand on the ground.'])

Summarization(only use English):
{{"video_name": "Abuse001_x264.mp4", "start_time": 4, "end_time": 18, "description": "Violent robbery, abuse, and assault were committed in the video.\
In a marble hall, a woman with silver short hair, dressed in white clothes and black pants, was reading newspapers at a wooden table in the center of the hall.\
Two criminals invaded from the bottom left corner. The first black man wearing a white T-shirt snatched the woman's red waist bag, and the second black man\
wearing a black T-shirt punched the woman in the head, causing her to fall to the ground and painfully cover her face before sitting up."}}


Video for you to summarize:
{}

Summarization(only use Englsh):
"""



video_summarization_prompt_normal = """
I need you to summarize a few footscripts about a video. 
I will give you 1 example from human annotators and then you straightly give your answer to the last video with the same format.
The answer should include video name, start time and end time which means the time count by second when the main events starts and ends, \
(Please give your best guess) and a highly summarized description of the main events. At the beginning of your description, you should first \
define there is no dangerous event detected in the video. Because what I should will be only normal videos. Then you should give details about this video.
Please follow the answer form and use ONLY ENGLISH.

Example 1:
('Abuse001_x264', ['00:00.0 00:05.3 ##A woman with short hair, slightly fat, wearing a white top and black pants stood in front of the table, \
picked up a book from the table, and opened it to read', '00:07.0 00:08.5 ##A man wearing a white shirt and black pants entered the house and \
walked towards the short-haired and fat woman in front who was reading a book.', '00:07.2 00:08.5 ##A man wearing a black shirt and black pants\
entered the house and walked towards the short-haired and fat woman in front who was reading a book.', \
"00:08.2 00:08.9 ##A man wearing a white shirt and black pants approached a short-haired, \
fat woman wearing a white shirt and black pants. When the woman was not paying attention, \
the man suddenly pulled out a piece of red cloth from the left side of the woman's body, then turned and ran away.",\
'00:08.9 00:11.2 ##A man in black clothes approached a short-haired, fat woman wearing a white top and black pants.\
When the woman turned back to the right, he punched the woman in the head and ran out. .',\
'00:08.9 00:11.2 ##The woman fell to the ground in pain, and the book in her hand fell to the ground when she fell.\
At the same time, she knocked the red wooden table in front of her crookedly. There were three things on the table.', \
'00:11.3 00:13.3 ##A woman with short hair and a fat figure wearing a white top and black pants fell to the ground,\
raised her right hand to touch her forehead, and hit her head on the wooden table leg.',\
'00:15.2 00:18.9 ##A woman with short hair, slightly fat, wearing a white top and black pants fell to the ground.\
She touched her forehead with her right hand and looked towards the direction of entering the house.\
She retracted her left leg, supported her left hand on the ground, and held her forehead with her right hand. Upper body.', \
'00:19.7 00:25.4 ##A woman with short hair, slightly fat, wearing a white top and black pants is sitting on the ground, \
leaning on the ground with her left hand, her right hand dropped from her head and placed on the knee of her right leg,\
her right leg is bent, she turns her head and kicks with her left leg Kneel down on the ground, stand up with your right hand on the ground.'])

Summarization(only use English):
{{"video_name": "Abuse001_x264.mp4", "start_time": 4, "end_time": 18, "description": "Violent robbery, abuse, and assault were committed in the video.\
In a marble hall, a woman with silver short hair, dressed in white clothes and black pants, was reading newspapers at a wooden table in the center of the hall.\
Two criminals invaded from the bottom left corner. The first black man wearing a white T-shirt snatched the woman's red waist bag, and the second black man\
wearing a black T-shirt punched the woman in the head, causing her to fall to the ground and painfully cover her face before sitting up."}}


Video for you to summarize:
{}

Summarization(only use Englsh):
"""

