'''

Code for creating and saving figures 16 and 17 from thesis

'''
import pandas as pd
import matplotlib.pyplot as plt
import glob

titles = ["Image", "Batch size", "Replay buffer size", "Stack size", "Gamma", "Learning rate", "Epsilon greedy"]
smoothing = 0.95
labels_all = ["Depth", "RGB", "Segmentation", 16, 32, 64, 5000, 25000, 100000, 1, 4, 10, 0.8, 0.9, 0.95, 0.99, 1e-3, 1e-4, 1e-5, 2000, 8000, 16000]


for configs in range(1,8):
    csv_files = glob.glob("data/"+str(configs)+"*.csv")
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

    if configs == 7: 
        csv_files_espilon = glob.glob("data/eps_"+str(configs)+"*.csv")
        # create empty list
        dataframes_list = []
        # append datasets into the list
        for i in range(len(csv_files_espilon)):
            temp_df = pd.read_csv(csv_files_espilon[i])
            dataframes_list.append(temp_df)
        
        for i in range(len(dataframes_list)):
            plt.plot(dataframes_list[i]["Step"],dataframes_list[i]["Value"])
            #labels.append(str(configs)+"."+str(i))

        plt.title(titles[configs-1] + " decay",size=12)
        plt.xlabel("Episode", size=11.5)
        plt.ylabel("Epsilon", size=11.5)
        plt.legend(labels)
        plt.savefig("figures/"+titles[configs-1] + " decay"+".pdf")
        plt.show()     
