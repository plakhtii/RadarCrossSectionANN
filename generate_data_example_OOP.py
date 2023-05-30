import numpy as np
import matplotlib.pyplot as plt
import math as m


class RadarCrossSection:
    def __init__(self):
        self.a = None
        self.lamda_value = None
        self.R = None
        self.r = None
        self.pi = np.pi
        self.Nphi = 720
        self.phi = np.linspace(0, 2 * np.pi, self.Nphi)
        self.create_points()
        self.th = list()
        self.rho = list()
        self.th_rho_append()
        self.Npy = self.r.shape[0]
        self.sigma0 = (4 * self.pi * (a ** 2 / lamda_value) ** 2)
        self.sigmaArr = (self.sigma0 / (self.Npy * self.Npy)) * np.ones((self.Npy, 1))
        self.Rarr = None
        self.NR = None
        self.sigma1 = None
        self.sigma1Far = None

    def cart_to_sph(self, x, y, z):
        rho = m.sqrt(x**2 + y**2)
        theta = m.atan2(y, x)
        return rho, theta

    def set_input_data(self, a: float, lamda_value: float, R: float):
        self.a = a
        self.lamda_value = lamda_value
        self.R = R
        self.Rarr = [self.lamda_value * val for val in [1, 2, 10, 100]]
        self.NR = len(self.Rarr)
        self.sigma1 = np.zeros((self.Nphi, self.NR))
        self.calculate_sigma_far()

    def sigmaN(self, theta1, theta2, phi1, phi2, sigma, r, lamda_value):
        # sigma is array (N,1); r is array (N,3)
        n1 = np.array([np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1)])
        n2 = np.array([np.sin(theta2) * np.cos(phi2), np.sin(theta2) * np.sin(phi2), np.cos(theta2)])
        delta = 2 * np.pi * (n1 + n2) / lamda_value
        N = len(sigma)
        res = np.sum(sigma)
        for i in range(N):
            for j in range(N):
                if i != j:
                    res = res + np.sqrt(sigma[i] * sigma[j]) * np.cos(np.dot(delta, (r[i, :] - r[j, :])))
        return res

    def SigmaS1(self, sigma0, r, R1, R2, phi1, theta1, phi2, theta2, wavelength):
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

    def getRCS(self, sigmaArr, points, lamda_value, phiArr, theta):
        Nphi = len(phiArr)
        rcs = np.zeros(Nphi)
        for i in range(Nphi):
            rcs[i] = self.sigmaN(theta, theta, phiArr[i], phiArr[i], sigmaArr, points, lamda_value)
        return rcs

    def create_points(self):
        self.r = np.array([[0, a*np.sqrt(2)/4, a*np.sqrt(2)/4],
                          [0, a*np.sqrt(2)/4, -a*np.sqrt(2)/4],
                           [0, -a*np.sqrt(2)/4, -a*np.sqrt(2)/4],
                           [0, -a*np.sqrt(2)/4, a*np.sqrt(2)/4]])

    def th_rho_append(self):
        for r_1, r_2, r_3 in zip(self.r[:, 1], self.r[:, 2], self.r[:, 0]):
            rho_tmp, th_tmp = self.cart_to_sph(r_1, r_2, r_3)
            self.th.append(th_tmp)
            self.rho.append(rho_tmp)

    def calculate_sigma_far(self):
        for i in range(self.NR):
            R = self.Rarr[i]
            self.sigma1[:, i] = self.SigmaS1(self.sigmaArr, self.r, R, R, self.phi, self.pi / 2, self.phi, self.pi / 2, self.lamda_value)
            self.sigma1Far = self.getRCS(self.sigmaArr, self.r, self.lamda_value, self.phi, self.pi / 2)

    def plot_sources(self):
        # Plotting 3D scatter plot of dots position
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.r[:, 0], self.r[:, 1], self.r[:, 2], c='green', marker='o')
        plt.show()

    def plot_rcs(self):
        # Plotting polar graph for far zone
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Plotting polar graph for far zone
        ax.plot(self.phi, self.sigma1Far / np.max(self.sigma1Far), linewidth=2)
        ax.legend(['Far zone'])

        # Plotting polar graphs for each R value
        legendData = ['Far zone']
        for i in range(self.NR):
            ax.plot(self.phi, self.sigma1[:, i] / np.max(self.sigma1[:, i]))
            legendData.append(str(self.Rarr[i] / lamda_value))

        ax.set_rticks([])  # Remove radial tick labels
        ax.legend(legendData)
        plt.show()


if __name__=='__main__':
    a = 1
    lamda_value = 1
    R = 1
    rcs_obj = RadarCrossSection()
    rcs_obj.set_input_data(a=a, lamda_value=lamda_value, R=R)
    rcs_obj.plot_sources()
    rcs_obj.plot_rcs()

