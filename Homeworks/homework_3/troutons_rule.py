import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from other_functions import ols_intercept, ols_slope, ols
from scipy.stats import t

#import values for T_B, H_v and a list of the classes 
df = pd.read_csv('trouton.csv')
T_B  = df['T_B (K)']
#convert H_v to units of J/mol (1 kcal/mol = 4.184 kJ/mol)
H_v = df['H_v (kcal/mol)'] * 4184
classes = df['Class']

class_list = list(set(classes))

#set up a list with a number of sublists equal to the number of unique classes
categories = []
for d in class_list:
    categories.append([])

#sort values of T_B and H_v by classes
for clas, T, H in zip(classes, T_B, H_v):
    values = categories[class_list.index(clas)]
    values.append((T,H))
    categories[class_list.index(clas)] = values

#plot each set of data points with their linear fits 
for i, set in enumerate(categories):
    #plot data points in each class as a scatter plot
    T = [x[0] for x in set]
    H = [x[1] for x in set]
    plt.scatter(T, H, label = class_list[i])

#calculate parameters for linear fit equation and plot it 
a,b = ols(T_B, H_v)
X = np.linspace(np.min(T_B) - 100, np.max(T_B) + 100, 500)
plt.plot(X, a * X + b, zorder = 0, linestyle = "--", alpha = 0.5, label = f"{class_list[i]} fit")

def confidence_intervals(a, b, x, y, confidence_level):
    residuals = y - a * x + b

    n_data_points = len(x)
    dof = n_data_points - 2 
    x_mean = np.mean(x)

    numerator = np.sum(residuals ** 2) / dof
    
    # denominators
    denominator_slope = np.sum((x - x_mean) ** 2)
    denominator_intercept = len(x) * np.sum((x - x_mean) ** 2)

    se_intercept = np.sqrt(numerator / denominator_intercept)
    se_slope = np.sqrt(numerator / denominator_slope)

    # Calculate the critical t-value
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, dof)

    return (critical_t_value * se_slope, critical_t_value * se_intercept)

#calculate confidence intervals
slope_interval, intercept_interval = confidence_intervals(a, b, T_B, H_v, 0.95)

#display relevant information
plt.annotate(r"$H_v = {:.3f} * T_B {:.3f}$".format(a,b), (1400,50000), fontsize = 8)
plt.annotate(r"a = {:.6f} $\pm$ {:.6f} $(J/mol-k)$".format(a,slope_interval), (1400,50000-10000), fontsize = 8)
plt.annotate(r"b = {:.6f} $\pm$ {:.6f} $(J/mol)$".format(b,intercept_interval), (1400,50000-20000), fontsize = 8)

#format plot 
plt.title('Troutons Rule')
plt.xlabel(r"$T_B$ (K)")
plt.ylabel(r'$H_v$ ($J/mol$)')
plt.legend()
plt.tight_layout()
plt.savefig('homework-3-1/troutons_rule.png', format = 'png')