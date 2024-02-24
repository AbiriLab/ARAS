'''

Code for creating and saving decaying epsilon for defense presentation

'''
import numpy as np
import matplotlib.pyplot as plt

epsilon = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.05,0.05,0.05])

steps = np.array([0,100,200,300,400,500,600,700,800,850,900,950,1000])


plt.figure()
plt.plot(steps,epsilon, linewidth= 2, color = "#002147")
plt.xlim([0, max(steps)])
plt.ylim([0, max(epsilon)+0.1])
plt.fill_between(steps,epsilon,color="#71C2FF", alpha=0.3)

#plt.savefig("figures/epsilon_defense.pdf")
plt.show()