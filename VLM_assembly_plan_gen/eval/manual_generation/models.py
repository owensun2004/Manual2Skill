from typing import Dict
from utils.data import Node
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
import numpy as np


class Model:

    def __init__(self):
        pass

    def __call__(self, feed_dict: Dict):
        pass


class SinglePartModel(Model):

    def __init__(self):
        super().__init__()

    def __call__(self, feed_dict: Dict):
        children = [Node(part_idx=i) for i in range(feed_dict['part_ct'])]
        return Node(children=children)


class SimilarityModel(Model):
    def __init__(self):
        super().__init__()
        self.min_n_cluster = 1
        self.max_n_cluster = 3
        assert self.min_n_cluster >= 1
        self.single_element_score = 0

    def __call__(self, feed_dict: Dict):
        part_idxs = list(range(feed_dict['part_ct']))
        return self.single_step(part_idxs, feed_dict['features'])

    def single_step(self, part_idxs, part_features):
        '''
        :param part_idxs:
               part_features: a list of part features to be clustered
        :return: the root Node for the input parts
        '''

        assert len(part_idxs) > 0
        if len(part_idxs) == 1:
            return Node(part_idx=list(part_idxs)[0])

        scores = []
        clustering_list = []
        for n_cluster in range(self.min_n_cluster, min(len(part_idxs), self.max_n_cluster) + 1):
            if n_cluster == 1:
                scores.append(self.score_single_cluster(part_features))
                clustering_list.append([None])
                continue
            # clustering = SpectralClustering(n_clusters=n_cluster,
            #                    assign_labels='discretize',
            #                    random_state=0).fit(part_features)
            clustering = KMeans(n_clusters=n_cluster,
                                random_state=0).fit(part_features)
            clustering_list.append(clustering)
            if max(clustering.labels_) + 1 < n_cluster:
                break
            scores.append(self.score_clustering(part_features, clustering.labels_))

        clustering_idx = np.argmax(scores)
        n_cluster = clustering_idx + self.min_n_cluster
        if n_cluster == 1:
            return Node(children=[Node(part_idx=i) for i in part_idxs])

        clustering = clustering_list[clustering_idx]

        scores = []
        for i in range(n_cluster):
            idxs = clustering.labels_ == i
            scores.append(self.score_single_cluster(part_features[idxs]))
        cluster_idx = np.argmax(scores)

        other_idxs = np.nonzero(clustering.labels_ != cluster_idx)[0]
        other_part_idxs = [part_idxs[idx] for idx in other_idxs]
        children = []
        for i in np.nonzero(clustering.labels_ == cluster_idx)[0]:
            children.append(Node(part_idx=part_idxs[i]))

        node_other = self.single_step(other_part_idxs, part_features[other_idxs])
        children.append(node_other)
        return Node(children=children[::-1])

    def score_single_cluster(self, features):
        '''
        :param features: (N, E)
        :return: a scalar scoring the cluster
        '''
        n = features.shape[0]
        if n == 0:
            import ipdb;
            ipdb.set_trace()
        if n == 1:
            return self.single_element_score
        # return the negative of the mean distance between pair of points
        return -((features[None] - features[:, None]) ** 2).sum() / (n * n - n)

    def score_clustering(self, features, labels):
        '''
        :param features: (N, E)
        :param labels: (N,)
        :return: a scalar scoring the clustering
        '''
        if features.shape[0] == max(labels) + 1:
            return -1
        return silhouette_score(features, labels, metric='euclidean')
