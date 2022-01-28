import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import pdb


class NeighborBase(object):
    def __init__(self, args, min_dist=1e-12):
        """
            k and alpha are the tuning parameters for the filtering,
            filtering: method of filtering. option are "none", "density",
            "uncertainty"
            min_dist: some small number to mitigate possible division by 0.
        """
        self.args = args
        self.dataset_name = args['ds']
        self.val_k = args['val_k']  # the number of neighbors queried
        self.filter_k = 10  # used for abnormal neighbor filtering
        self.min_dist = min_dist  # used for abnormal neighbor filtering
        self.filtering = args['filtering']  # filtering method
        self.alpha = args['TS_alpha']  # percentage of abnormal neighbors filtered

        self.by_class = args['kdtree_by_class']

        self.temp = args['similarity_T']
        self.sim_kernel = (self.temp is not None)
        print("using similarity kernels with temperature {}".format(self.temp))

    def filter_by_density(self, X):
        """Filter out points with low kNN density.
        Args:
        X: an array of sample points.

        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        original points of kNN density.
        """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, self.filter_k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X, y):
        """Filter out points with high label disagreement amongst its kNN neighbors.
        Args:
        X: an array of sample points.

        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        samples with highest disagreement amongst its k nearest neighbors.
        """
        neigh = KNeighborsClassifier(n_neighbors=self.filter_k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        confidence = confidence[range(y.shape[0]), y]
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, X, y):
        """use training data to build a KD-Tree.

        WARNING: assumes that the labels are 0-indexed (i.e.
        0, 1,..., n_labels-1).

        Args:
        X: an array of sample points.
        y: corresponding labels.
        """
        self.n_labels = np.max(y) + 1

        if self.by_class:
            self.kdtree = [None] * self.n_labels
            if self.filtering == "uncertainty":
                X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
            # in the training dataset, for every label, build a KD-Tree
            for label in range(self.n_labels):
                if self.filtering == "none":
                    X_to_use = X[np.where(y == label)[0]]
                elif self.filtering == "density":
                    X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                elif self.filtering == "uncertainty":
                    X_to_use = X_filtered[np.where(y_filtered == label)[0]]

                if len(X_to_use) == 0:
                    print("Filtered too much or missing examples from a label! Please lower alpha or check data.")
                self.kdtree[label] = KDTree(X_to_use)

        else:
            if self.filtering == "none":
                X_to_use = X
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X)
            elif self.filtering == "uncertainty":
                X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
                X_to_use = X_filtered

            if len(X_to_use) == 0:
                print("Filtered too much or missing examples from a label! Please lower alpha or check data.")
            else:
                ratio = len(X_to_use) / len(X)
                print("{:.2%} has been filtered out.".format(1 - ratio))
            self.kdtree = KDTree(X_to_use)

        print("Building KDTree done.")

    def get_neighbor_vec(self, X):
        if self.by_class:
            d = np.tile(None, (X.shape[0], self.n_labels, self.val_k))
            # for every label, compute every test sample's NN?
            for label_idx in range(self.n_labels):
                try:
                    d[:, label_idx, :] = self.kdtree[label_idx].query(X, k=self.val_k)[0][:, :self.val_k]
                except:
                    # one class's number of samples is less than self.val_k
                    k = self.kdtree[label_idx].data.shape[0]
                    d[:, label_idx, :k] = self.kdtree[label_idx].query(X, k=k)[0][:, :k]
                    print("one class's number of samples is less than self.val_k")
            d = d.reshape((d.shape[0], -1)).astype(np.float)
        else:
            num_neighbors = self.n_labels * self.val_k
            d = np.tile(None, (X.shape[0], num_neighbors))
            d[:, :] = self.kdtree.query(X, k=num_neighbors)[0][:, :num_neighbors]
            d = d.reshape((d.shape[0], -1)).astype(np.float)

        if self.sim_kernel:
            d = np.exp(-d.astype(np.float) / self.temp)
        return d

