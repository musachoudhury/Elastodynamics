# import numpy as np
# from math import cos, tan, log, pi, atan

# w = 3.0
# l = 3.0

# def sec(x):
#     return 1/cos(x)

# def f(x):
#     return sec(x)*tan(x) + log(abs(sec(x)+tan(x)))

# phi = atan(l/w)
    
# I = w**3/6*(f(phi)-f(0)+f(pi/2)-f(phi))
# print(I)
#print(1200/20.66)

import numpy as np
import matplotlib.pyplot as plt 


x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)


xv, yv = np.meshgrid(x, y, indexing='ij')

print(xv)  
# V = np.array([[1, 1], [-2, 2], [4, -7]])
# origin = np.array([[0, 0, 0], [0, 0, 0]])



# plt.quiver(*origin, V[:, 0], V[:, 0], color=['r', 'b', 'g'], scale=21)
# plt.show()