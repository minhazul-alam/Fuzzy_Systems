# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def fuzzifier(x, F1, F2):
    f_x = []
    for x_val in x:
        y = 1./((F2/x_val)**F1 + 1)
        f_x.append(y)
    return f_x

x = np.arange(-1.0, 15, 0.5)
y1 = fuzzifier(x, 100, 50)
y2 = fuzzifier(x, 10, 50)
y3 = fuzzifier(x, 4, 50)
y4 = fuzzifier(x, 2, 50)
y5 = fuzzifier(x, 1, 50)
plt.plot(x, y1, x, y2, x, y3, x, y4, x, y5)

y1 = fuzzifier(x, 4, 30)
y2 = fuzzifier(x, 4, 40)
y3 = fuzzifier(x, 4, 50)
y4 = fuzzifier(x, 4, 60)
y5 = fuzzifier(x, 4, 70)
plt.plot(x, y1, x, y2, x, y3, x, y4, x, y5)




x_B = np.arange(3, 15, 0.5)
y_B = np.zeros(len(x_B))
for i, x in enumerate(x_B):
    y_B[i] = (4-(x+1)**0.5)/8

plt.xlabel('x')
plt.ylabel('mu(x)')
plt.title('Fuzzy Set - B')
plt.plot(x_B, y_B)


plt.xlabel('x')
plt.ylabel('mu(x)')
plt.title('Fuzzy Set - B')
plt.show()
t = np.arange(0., 5., 0.2)
plt.plot(t, t, t, t**2)
plt.show()









