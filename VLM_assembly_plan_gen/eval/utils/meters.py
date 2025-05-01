from collections import defaultdict


class Meters:
    def __init__(self):
        self.sum_d = defaultdict(int)
        self.n_d = defaultdict(int)

    def update(self, name, v, n=1):
        self.sum_d[name] += v
        self.n_d[name] += n

    def avg(self, name):
        return self.sum_d[name] / self.n_d[name]

    def avg_dict(self):
        d = {}
        for k in self.sum_d.keys():
            d[k] = self.sum_d[k] / self.n_d[k]
        return d

    def merge_from(self, m, ks=None):
        if ks is None:
            ks = m.sum_d.keys()
        for k in ks:
            self.sum_d[k] += m.sum_d[k]
            self.n_d[k] += m.n_d[k]
