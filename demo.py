import json
from pathlib import Path
from starter.tree_train import fit, objective, predict
from starter.action import closest
p=Path(__file__).parent
case=json.loads((p/'fixtures/xor.json').read_text())
t=fit(case['rows'],case['costs'],case['thresholds'],case['max_depth'],case['alpha'])
print(json.dumps({'model':t,'objective':objective(t,case['rows'],case['costs'],case['alpha']),
                 'ordinary_action':closest({'feature':0,'threshold':1,'left':{'label':0},'right':{'label':1}},[0,0],1,[[0,3],[0,3]],[1,2])},ensure_ascii=False))
