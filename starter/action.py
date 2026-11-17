"""Working closest-leaf explanation for ordinary bounded integer features.
The old interface knows boxes and symmetric L1 costs, not joint action rules.
"""
from .tree_train import predict

def closest(tree, current, target, bounds, weights):
    results=[]
    def visit(t,box):
        if 'label' in t:
            if t['label']!=target:return
            point=[min(max(x,l),u) for x,(l,u) in zip(current,box)]
            if any(l>u for l,u in box):return
            value=sum(w*abs(x-y) for w,x,y in zip(weights,current,point))
            results.append((value,point));return
        f=t['feature'];th=t['threshold']
        a=[list(z) for z in box];a[f][1]=min(a[f][1],th)
        c=[list(z) for z in box];c[f][0]=max(c[f][0],th+1)
        visit(t['left'],a);visit(t['right'],c)
    visit(tree,[list(z) for z in bounds])
    return None if not results else min(results,key=lambda z:(z[0],z[1]))
