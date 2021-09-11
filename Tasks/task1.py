import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = pd.Series([69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76])
y = pd.Series([153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140,  150, 165, 185, 210, 220])

print("Variance:", x.var())
# x.plot.density()

mean = np.mean(x)
std = np.std(x)
variance = x.var()

axis = np.linspace(mean - 5 * std, mean + 5 * std, 300)
f = np.exp(-np.square(axis-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

plt.plot(axis, norm.pdf(axis, mean, std))

plt.show()

