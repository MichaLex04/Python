import matplotlib.pyplot as plt
import seaborn as sns

tips=sns.load_dataset("tips")
print("Propina minima",tips["tip"].min())
print("Propina minima",tips["tip"].max())
plt.hist(tips["tip"],edgecolor="black")
plt.show()
