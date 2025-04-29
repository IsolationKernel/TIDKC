from sklearn.kernel_approximation import Nystroem
import numpy as np


def gdk(list_of_distributions, gma1):
    gdk_nys = Nystroem(gamma=gma1, n_components=200)
    D_idx = [0]  # index of each distributions
    alldata = []
    n = len(list_of_distributions)
    for i in range(1, n + 1):
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        alldata += list_of_distributions[i - 1].tolist()
    alldata = np.array(alldata)

    all_gdkmap1 = gdk_nys.fit_transform(alldata)

    gdkmap1 = []
    for i in range(n):
        gdkmap1.append(np.sum(all_gdkmap1[D_idx[i]:D_idx[i + 1]], axis=0) / (D_idx[i + 1] - D_idx[i]))
    gdkmap1 = np.array(gdkmap1)
    gdk = np.zeros((n,n))
    '''for i in range(n):
        for j in range(n):
            gdk[i][j] = np.dot(gdkmap1[i],gdkmap1[j].T)'''
    return gdkmap1