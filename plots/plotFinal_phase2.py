'''

Code for creating and saving figure 18 from thesis

'''
import pandas as pd
import matplotlib.pyplot as plt
import glob

smoothing = 0.9
smoothing_background = 0.0
titles = ["final_model_phase_2"]

for configs in range(1):
    csv_files = glob.glob("data/phase2/"+"phase2_final.csv")
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
        smooth_background = dataframes_list[i].ewm(alpha=(1 - smoothing_background)).mean()
        plt.plot(dataframes_list[i]["Step"],smooth_background["Value"], alpha=0.2)
        plt.plot(dataframes_list[i]["Step"],smooth["Value"],color="#002147")


    plt.xlabel("Episode", size=12.5)
    plt.ylabel("Ten episode average reward", size=12.5)
    plt.savefig("figures/final/"+titles[configs]+".pdf")
    plt.show()
