import os
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

plt.plot(x=[1,2], y=[3,4])
plt.xlabel('$\ell_\infty$ distance')
plt.savefig('test.pdf')
