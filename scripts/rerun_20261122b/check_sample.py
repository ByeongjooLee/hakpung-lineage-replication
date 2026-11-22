import json, sys, io
N = sys.argv[1]; N2 = sys.argv[2]
old = json.load(io.open(N + '/_genealogy_20261122/site/data/genealogy-library.json', encoding='utf-8'))
new = json.load(io.open(N2 + '/_genealogy/site/data/genealogy-library.json', encoding='utf-8'))
meta = json.load(io.open(N + '/bge_m3/vectors_meta.json', encoding='utf-8'))
ek = 'edges' if 'edges' in new else 'links'
def key(e): return (e['source'], e.get('thesis_id') or e.get('thesis') or e.get('id'))
print('edge keys sample', {k: (str(v)[:40]) for k, v in list(new[ek][0].items())[:8]})
newset = {}
for e in new[ek]: newset.setdefault(e['source'], set()).add(e['target'])
oldset = {}
for e in old[ek]: oldset.setdefault(e['source'], set()).add(e['target'])
profs = list(meta['professors']) if isinstance(meta['professors'], dict) else [p if isinstance(p, str) else p.get('professor_id') for p in meta['professors']]
nn = {n['id'] for n in new['nodes']}
print('sample profs', len(profs), 'missing in new', sum(1 for p in profs if p not in nn))
diff = [(p, len(oldset.get(p, set()) ^ newset.get(p, set()))) for p in profs if oldset.get(p, set()) != newset.get(p, set())]
print('profs whose target set changed', diff[:10])
chg = [(e['source'][-12:], e['target'][-25:]) for e in new[ek] if e['target'] not in {x['target'] for x in old[ek]}]
print('new targets', chg[:10])
def st(g): 
    from collections import Counter
    return Counter((e.get('student_identity') or {}).get('status') for e in g[ek])
print('old status', st(old)); print('new status', st(new))
