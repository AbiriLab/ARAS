'''

Code for reading the training time of the different settings

'''

import datetime as dt
from datetime import datetime
import glob

date_format = '%Y-%m-%d %H:%M:%S.%f'
date_format_hours = "%H:%M:%S.%f"

#system = ["rob", "sima", "ali"]
system = ["final"]


for content in range(5):
    #log_files = glob.glob("phase_data"+ system[content] +"/phase2/logs/learnDQN"+"*.log")
    #log_files = glob.glob("phase_data/phase2/logs/learnDQN"+"*_old.log")
    log_files = glob.glob("phase_data/phase2/logs/learnDQN"+"*.log")
    print("Logfiles from system of " + system[content] +":" )
    content_files = []
    for i in range(len(log_files)):
        log_file = open(log_files[i]) 
        content = log_file.readlines()
        time_add = dt.timedelta(seconds=0)

        for line in range(4,len(content)):
            timestring_before = datetime.strptime(content[line-1][0:26],date_format)
            timestring_current = datetime.strptime(content[line][0:26],date_format) 
            timedifference = (timestring_current-timestring_before).total_seconds() 
            if timedifference < 0:
                # print(timestring_before,timestring_current,timedifference) 
                # print(line)
                # print(content[line][43:55])
                time_add = time_add + dt.timedelta(seconds=float(content[line][43:55]))
                #print(time_add)
                #print(type(time_add))

        timestring_first = datetime.strptime(content[4][0:26],date_format)
        timestring_last = datetime.strptime(content[-1][0:26],date_format)  
        total_length = timestring_last - timestring_first 
        total_length = total_length + time_add
        print(total_length)
        content_files.append(total_length)
    
    #print(content_files)
    print("Average:" +str(sum(content_files, dt.timedelta())/len(content_files)))

    print("______________________________________")