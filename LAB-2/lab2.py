import skfuzzy
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 1.0001, 0.0001)
y = np.arange(0, 1.0001, 0.0001)
triangle_f = skfuzzy.trimf(x, [0.2, 0.4, 0.6])
trap_f = skfuzzy.trapmf(x, [0.2, 0.3, 0.4, 0.5])

gauss_f = skfuzzy.gaussmf(x, 0.5, 0.1)
two_side_gauss_f = skfuzzy.gauss2mf(x, 0.5, 0.1, 0.5, 0.2)
j = np.arange(0, 10, 0.1)
bell_f = skfuzzy.gbellmf(j, 2, 2, 5)

sig_f = skfuzzy.sigmf(x, 0.5, 15)
p_sig_f = skfuzzy.psigmf(x, 0.3, 15, 0.7, 15)
d_sig_f = skfuzzy.dsigmf(x, 0.3, 15, 0.7, 40)

pi_f = skfuzzy.pimf(x, 0.1, 0.3, 0.5, 0.9)
s_f = skfuzzy.smf(x, 0.2, 0.8)
z_f = skfuzzy.zmf(x, 0.2, 0.8)
plt.plot(x, triangle_f)
plt.show()
plt.plot(x, trap_f)
plt.show()
plt.plot(x, gauss_f)
plt.show()
plt.plot(x, two_side_gauss_f)
plt.show()
plt.plot(j, bell_f)
plt.show()
plt.plot(x, sig_f)
plt.show()
plt.plot(x, p_sig_f)
plt.show()
plt.plot(x, d_sig_f)
plt.show()
plt.plot(x, pi_f)
plt.show()
plt.plot(x, s_f)
plt.show()
plt.plot(x, z_f)
plt.show()



g1 = skfuzzy.gaussmf(x, 0.4, 0.2)
g2 = skfuzzy.gaussmf(y, 0.7, 0.2)
_, g3 = skfuzzy.fuzzy_and(x, g1, y, g2)
_, g4 = skfuzzy.fuzzy_or(x, g1, y, g2)
g5 = skfuzzy.fuzzy_not(g1)

g6 = g1 + g2 - g1 * g2
g7 = g1 * g2

plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, g3)
plt.show()

plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, g4)
plt.show()

plt.plot(x, g1)
plt.plot(x, g5)
plt.show()

plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, g6)
plt.show()
#
plt.plot(x, g1)
plt.plot(x, g2)
plt.plot(x, g7)
plt.show()




