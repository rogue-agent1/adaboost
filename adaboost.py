#!/usr/bin/env python3
"""AdaBoost classifier — zero dependencies."""
import math, random, sys
from collections import Counter

class DecisionStump:
    def __init__(self):
        self.feature = 0; self.threshold = 0; self.polarity = 1; self.alpha = 0
    def predict(self, X):
        return [self.polarity if x[self.feature] <= self.threshold else -self.polarity for x in X]

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators; self.stumps = []
    def fit(self, X, y):
        n = len(X); w = [1/n]*n; self.stumps = []
        for _ in range(self.n_estimators):
            stump = self._best_stump(X, y, w)
            preds = stump.predict(X)
            err = sum(w[i] for i in range(n) if preds[i] != y[i])
            err = max(err, 1e-10); err = min(err, 1-1e-10)
            stump.alpha = 0.5 * math.log((1-err)/err)
            w = [w[i]*math.exp(-stump.alpha*y[i]*preds[i]) for i in range(n)]
            s = sum(w); w = [x/s for x in w]
            self.stumps.append(stump)
    def _best_stump(self, X, y, w):
        best, best_err = DecisionStump(), float('inf')
        n_feat = len(X[0])
        for f in range(n_feat):
            vals = sorted(set(x[f] for x in X))
            for t in vals:
                for pol in [1, -1]:
                    preds = [pol if x[f] <= t else -pol for x in X]
                    err = sum(w[i] for i in range(len(X)) if preds[i] != y[i])
                    if err < best_err:
                        best_err = err; best.feature = f; best.threshold = t; best.polarity = pol
        return best
    def predict(self, X):
        scores = [0]*len(X)
        for s in self.stumps:
            preds = s.predict(X)
            for i in range(len(X)): scores[i] += s.alpha * preds[i]
        return [1 if s > 0 else -1 for s in scores]

def test():
    random.seed(42)
    X = [[random.gauss(1,1), random.gauss(1,1)] for _ in range(50)] + \
        [[random.gauss(-1,1), random.gauss(-1,1)] for _ in range(50)]
    y = [1]*50 + [-1]*50
    clf = AdaBoost(n_estimators=10); clf.fit(X, y)
    preds = clf.predict(X)
    acc = sum(1 for p,t in zip(preds,y) if p==t)/len(y)
    print(f"AdaBoost accuracy: {acc:.1%} ({len(clf.stumps)} stumps)")
    assert acc > 0.7, f"Accuracy too low: {acc}"
    print("All tests passed!")

if __name__ == "__main__":
    test() if len(sys.argv) < 2 or sys.argv[1] == "test" else print("Usage: adaboost.py [test]")
