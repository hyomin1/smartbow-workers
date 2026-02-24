class SuppressionCache:
    def __init__(self, max_hits=3, dist_thresh=20.0):
        self.max_hits = max_hits
        self.dist_thresh = dist_thresh
        self.hits = []

    def reset(self):
        self.hits.clear()

    def _dist(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def push_and_check(self, point):
        if not self.hits:
            self.hits.append(point)
            return False

        if self._dist(point, self.hits[-1]) <= self.dist_thresh:
            self.hits.append(point)
        else:
            self.hits.clear()
            self.hits.append(point)

        return len(self.hits) >= self.max_hits
