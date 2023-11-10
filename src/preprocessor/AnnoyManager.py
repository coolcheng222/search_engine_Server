from annoy import AnnoyIndex
import random
class BaseAnnoyManager:
    def __init__(self, dim):
        self.index = AnnoyIndex(dim, metric='angular')
        self.current_index = 0

    def add_vectors(self, vectors):
        for vector in vectors:
            self.index.add_item(self.current_index, vector.numpy())
            self.current_index += 1

    def build_index(self, n_trees=10):
        self.index.build(n_trees)

    def find_nearest_neighbors(self, query_vector, k=1):
        indices, distances = self.index.get_nns_by_vector(query_vector.numpy(), k, include_distances=True)
        return indices, distances
    def save_index(self, filename):
        self.index.save(filename)

    def load_index(self, filename):
        self.index.load(filename)
