import numpy as np
import matplotlib.pyplot as plt
import math as m


def cart2sph(x, y, z):
    rho = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)
    return rho, theta


def sigmaN(theta1, theta2, phi1, phi2, sigma, r, _lambda):
    # sigma is array (N,1); r is array (N,3)
    n1 = np.array([np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1)])
    n2 = np.array([np.sin(theta2) * np.cos(phi2), np.sin(theta2) * np.sin(phi2), np.cos(theta2)])
    delta = 2 * np.pi * (n1 + n2) / _lambda
    N = len(sigma)
    res = np.sum(sigma)
    for i in range(N):
        for j in range(N):
            if i != j:
                res = res + np.sqrt(sigma[i] * sigma[j]) * np.cos(np.dot(delta, (r[i, :] - r[j, :])))
    return res


def SigmaS1(sigma0, r, R1, R2, phi1, theta1, phi2, theta2, wavelength):
    # Kogerentnyy vypadok bez DR blyzhnia zona
    sigma1 = np.zeros(phi1.shape)

    for k in range(phi1.shape[0]):
        for j in range(r.shape[0]):
            for i in range(r.shape[0]):
                n1 = np.array([np.sin(theta1)*np.cos(phi1[k]), np.sin(theta1)*np.sin(phi1[k]), np.cos(theta1)])
                n2 = np.array([np.sin(theta2)*np.cos(phi2[k]), np.sin(theta2)*np.sin(phi2[k]), np.cos(theta2)])

                if i != j:
                    sigma1[k] += np.sqrt(sigma0[i] * sigma0[j]) * (
                        np.cos(2*np.pi/wavelength * np.sqrt(np.dot(2*r[i,:] - R1*n1 - R2*n2, 2*r[i,:] - R1*n1 - R2*n2))) * np.cos(2*np.pi/wavelength * np.sqrt(np.dot(2*r[j,:] - R1*n1 - R2*n2, 2*r[j,:] - R1*n1 - R2*n2))) +
                        np.sin(2*np.pi/wavelength * np.sqrt(np.dot(2*r[i,:] - R1*n1 - R2*n2, 2*r[i,:] - R1*n1 - R2*n2))) * np.sin(2*np.pi/wavelength * np.sqrt(np.dot(2*r[j,:] - R1*n1 - R2*n2, 2*r[j,:] - R1*n1 - R2*n2)))
                    )

    sigma = sigma1
    return sigma

def getRCS(sigmaArr, points, _lambda, phiArr, theta):
    Nphi = len(phiArr)
    rcs = np.zeros(Nphi)
    for i in range(Nphi):
        rcs[i] = sigmaN(theta, theta, phiArr[i], phiArr[i], sigmaArr, points, _lambda)
    return rcs

R = 1
Nphi = 720
phi = np.linspace(0, 2*np.pi, Nphi)
a = 1
lamda_value = 1
pi = 3.1415


r = np.array([[0, a*np.sqrt(2)/4, a*np.sqrt(2)/4],
              [0, a*np.sqrt(2)/4, -a*np.sqrt(2)/4],
              [0, -a*np.sqrt(2)/4, -a*np.sqrt(2)/4],
              [0, -a*np.sqrt(2)/4, a*np.sqrt(2)/4]])

th = list()
rho = list()

for r_1, r_2, r_3 in zip(r[:, 1], r[:, 2], r[:, 0]):
    rho_tmp, th_tmp = cart2sph(r_1, r_2, r_3)
    th.append(th_tmp)
    rho.append(rho_tmp)

Npy = r.shape[0]
sigma0 = (4*pi*(a**2/lamda_value)**2)
sigmaArr = (sigma0 / (Npy * Npy)) * np.ones((Npy, 1))
Rarr = [lamda_value * val for val in [1, 2, 10, 100]]

NR = len(Rarr)
sigma1 = np.zeros((Nphi, NR))

for i in range(NR):
    R = Rarr[i]

    sigma1[:, i] = SigmaS1(sigmaArr, r, R, R, phi, np.pi / 2, phi, np.pi / 2, lamda_value)

    sigma1Far = getRCS(sigmaArr, r, lamda_value, phi, np.pi / 2)


# Plotting polar graph for far zone
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Plotting polar graph for far zone
ax.plot(phi, sigma1Far / np.max(sigma1Far), linewidth=2)
ax.legend(['Far zone'])

# Plotting polar graphs for each R value
legendData = ['Far zone']
for i in range(NR):
    ax.plot(phi, sigma1[:, i] / np.max(sigma1[:, i]))
    legendData.append(str(Rarr[i] / lamda_value))

ax.set_rticks([])  # Remove radial tick labels
ax.legend(legendData)

# plt.show()

# Plotting 3D scatter plot of dots position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r[:, 0], r[:, 1], r[:, 2], c='green', marker='o')

plt.show()
