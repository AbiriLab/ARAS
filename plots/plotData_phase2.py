'''

Code for creating and saving figures 19 from thesis

'''

import pandas as pd
import matplotlib.pyplot as plt
import glob

titles = ["Reward function"]
smoothing = 0.96
smooting_steps = 0.995
labels_all = ["R1", "R2", "R3"]


for configs in range(1):
    csv_files = glob.glob("data/phase2/R*.csv")
    # create empty list
    dataframes_list = []
 
    # append datasets into the list
    for i in range(len(csv_files)):
        temp_df = pd.read_csv(csv_files[i])
        dataframes_list.append(temp_df)

    #print(dataframes_list)

    labels = []
    for i in range(len(dataframes_list)):
        smooth = dataframes_list[i].ewm(alpha=(1 - smoothing)).mean()
        plt.plot(dataframes_list[i]["Step"],smooth["Value"])
        if configs < 6:
            labels.append(str(labels_all[3*(configs-1)+i]))
        else:
            labels.append(str(labels_all[3*(configs-1)+i+1]))
        

    plt.title(titles[configs-1],size=12)
    plt.xlabel("Episode",size=11.5)
    plt.ylabel("Ten episode average reward",size=11.5)
    plt.legend(labels)
    plt.savefig("figures/"+titles[configs-1]+".pdf")
    plt.show()

    if configs == 0: 
        csv_files_steps = glob.glob("data/phase2/steps_*.csv")
        # create empty list
        dataframes_list = []
        # append datasets into the list
        for i in range(len(csv_files_steps)):
            temp_df = pd.read_csv(csv_files_steps[i])
            dataframes_list.append(temp_df)
        
        for i in range(len(dataframes_list)):
            smooth_steps = dataframes_list[i].ewm(alpha=(1 - smooting_steps)).mean()
            plt.plot(dataframes_list[i]["Step"],smooth_steps["Value"])
            #labels.append(str(configs)+"."+str(i))

        plt.title(titles[configs-1],size=12)
        plt.xlabel("Episode", size=11.5)
        plt.ylabel("Steps until grasp", size=11.5)
        plt.legend(labels)
        plt.savefig("figures/"+titles[configs-1] + "_steps"+".pdf")
        plt.show()     
