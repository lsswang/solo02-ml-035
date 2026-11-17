"""Working greedy numeric CART used by the current classifier."""
import math

def predict(tree, row):
    while 'label' not in tree:
        value = row[tree['feature']]
        left = tree.get('missing_left', True) if value is None else value <= tree['threshold']
        tree = tree['left' if left else 'right']
    return tree['label']

def fit(rows, costs, thresholds, max_depth=2, alpha=0.05):
    def leaf(ids):
        losses=[math.fsum(costs[i][y] for i in ids) for y in (0,1)]
        y=min(range(2),key=lambda k:(losses[k],k))
        return {'label':y},losses[y]
    def grow(ids, depth):
        base,loss=leaf(ids)
        if not depth: return base
        best=None
        for f, ts in enumerate(thresholds):
            for t in ts:
                a=[i for i in ids if rows[i][f] is None or rows[i][f]<=t]
                c=[i for i in ids if i not in a]
                if not a or not c: continue
                score=leaf(a)[1]+leaf(c)[1]+alpha
                if score<loss-1e-12 and (best is None or score<best[0]): best=(score,f,t,a,c)
        if best is None:return base
        _,f,t,a,c=best
        return {'feature':f,'threshold':t,'missing_left':True,'left':grow(a,depth-1),'right':grow(c,depth-1)}
    return grow(list(range(len(rows))),max_depth)

def objective(tree,rows,costs,alpha):
    def splits(t):return 0 if 'label' in t else 1+splits(t['left'])+splits(t['right'])
    return math.fsum(costs[i][predict(tree,x)] for i,x in enumerate(rows))+alpha*splits(tree)
