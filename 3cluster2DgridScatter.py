import numpy as np
import matplotlib.pyplot as plt

# Create the data:
D = 2 # dimensionality of the input
Nclass = 500

X1 = np.random.randn(Nclass, D) + np.array([0 ,-2]) #center on x=0, y=-2
X2 = np.random.randn(Nclass, D) + np.array([2 , 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X  = np.vstack([X1,X2,X3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

# Plot the data:
plt.scatter(X[:,0], X[:1], c=Y, s=100, alpha=0.5)
plt.show()
