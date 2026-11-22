# -*- coding: utf-8 -*-
"""새 표본의 간선별 유사도·순위 (prelim_similarity.py 215~242행과 같은 식; 태그·창 열은 뺌). 실행: py -X utf8 edges_similarity_new.py <표현 폴더>"""
import sys, json, csv
from pathlib import Path
import numpy as np
d = Path(sys.argv[1]); z = np.load(d / 'vectors.npz', allow_pickle=True); meta = json.load(open(d / 'vectors_meta.json', encoding='utf-8'))
S = z['stu'].astype(np.float64); P = z['prof'].astype(np.float64); pids = [str(x) for x in z['pids']]; pidx = {p: i for i, p in enumerate(pids)}
punis = [set(meta['professors'][p]['universities']) for p in pids]; rows = []
for i, e in enumerate(meta['edges']):
    sims = P @ S[i]; j = pidx[e['professor_id']]; uni = e['university']
    r = {'edge_id': e['edge_id'], 'professor_id': e['professor_id'], 'professor': meta['professors'][e['professor_id']]['name'], 'student': e['student'], 'thesis_id': e['thesis_id'], 'year': e['year'], 'university': uni, 'identity': e['identity'],
         'emb_sim': round(float(sims[j]), 4), 'emb_mean_other_profs': round(float(np.delete(sims, j).mean()), 4), 'emb_rank_all': int((sims > sims[j]).sum()) + 1, 'n_prof_all': len(pids), 'emb_rank_same_uni': '', 'n_prof_same_uni': '', 'emb_z_same_uni': ''}
    same = [k for k in range(len(pids)) if uni and uni in punis[k]]
    if len(same) >= 2 and j in same:
        ss = sims[same]; r['emb_rank_same_uni'] = int((ss > sims[j]).sum()) + 1; r['n_prof_same_uni'] = len(same)
        others = [x for k, x in zip(same, ss) if k != j]; r['emb_z_same_uni'] = round((sims[j] - float(np.mean(others))) / (float(np.std(others)) + 1e-9), 3)
    rows.append(r)
with open(d / 'edges_similarity.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
rk = [(r['emb_rank_same_uni'], r['n_prof_same_uni']) for r in rows if r['emb_rank_same_uni'] != '']
s = {'n_edges': len(rows), 'mean_sim_advisor': float(np.mean([r['emb_sim'] for r in rows])), 'mean_sim_other': float(np.mean([r['emb_mean_other_profs'] for r in rows])),
     'n_same_uni': len(rk), 'top1_same_uni': float(np.mean([k == 1 for k, n in rk])), 'chance_top1_same_uni': float(np.mean([1 / n for k, n in rk])),
     'top1_all': float(np.mean([r['emb_rank_all'] == 1 for r in rows])), 'n_prof_all': len(pids), 'median_rank_all': float(np.median([r['emb_rank_all'] for r in rows]))}
json.dump(s, open(d / 'edges_similarity_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(json.dumps(s, ensure_ascii=False))
