import numpy as np
import matplotlib.pyplot as plt

from models.KMeans import KMeans

########################################################################################################################


def four_cluster_gen(m, d):
    arrays = [np.random.randn(m, 2) for i in range(4)]
    arrays[1][:, 0] += d
    arrays[2][:, 1] += d
    arrays[3] += d
    X = np.vstack(tuple((ar for ar in arrays)))
    return X


########################################################################################################################

X = four_cluster_gen(1000, 5)

recorded_losses = []
for i in range(20):
    print(F'model {i}')
    mdl = KMeans(num_clusters=4, max_iterations=30)
    mdl.fit(X)
    recorded_losses.append(mdl.losses)

print('plotting')
for loss in recorded_losses:
    plt.plot(loss, alpha=0.5)
plt.show()
