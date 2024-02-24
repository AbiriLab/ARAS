'''

Code for merging single video clips from log_video_phase1.py,  log_video_phase2.py, video_final.py together

'''


# putting videos together
from moviepy.editor import *
import os
from natsort import natsorted

L = []

#choose which phase
phase = "phase2" # [phase1, phase2, final]

for root, dirs, files in os.walk("video/"+phase+"/clips/"):

    #files.sort()
    files = natsorted(files)
    for file in files:
        if os.path.splitext(file)[1] == '.mp4':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("video/"+phase+"/final/"+phase+"_final.mp4", fps=24, remove_temp=False)