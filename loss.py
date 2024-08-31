import matplotlib.pyplot as plt
import numpy as np

def reg(x, p):
    return np.abs(x) ** p

x, y = np.meshgrid(np.linspace(-4, 4, num=100), np.linspace(-4, 4, num=100))
pars = [0.5, 1, 2]

plt.figure(figsize=(15,4))

for (i, p) in enumerate(pars, start=1):
    z = reg(x, p) + reg(y, p)
    plt.subplot(1, len(pars), i)
    plt.contourf(x,y,z)
    plt.title(r"$|\beta_1|^{%s} + |\beta_2|^{%s}$" % (p, p))
    
plt.show()
