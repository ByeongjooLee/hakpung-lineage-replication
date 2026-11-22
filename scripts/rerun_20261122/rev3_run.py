# -*- coding: utf-8 -*-
"""역산 다듬기.
 (가) 박사논문만 묶기 — 임베딩 셋 × 설정 일곱(HDBSCAN 여섯 + k-means)으로 되풀이해 결과가 설정에 기대는지 본다.
 (나) 교수를 함께 넣어 묶기 — 박사논문 1,593편 + 교수 295명. 제자 논문과 같은 묶음에 드는 교수가 누구인지(지도교수·동문·3촌·조부모·계보 밖 같은 대학)를 센다.
묶을 때는 대학·지도교수·계보 정보를 쓰지 않는다. 관계 표지는 공개 파생표(2026-11-10 판 학맥)에서 읽어 묶은 뒤에만 쓴다.
실행: <venv python> -X utf8 rev2_run.py <big> <<SITE>> <계보 폴더> <색인 text_sp> <공개_파생표 폴더> <out>
"""
import sys, json, re, glob, csv, gzip, collections, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
big, repo, gen, index, tables, out = (Path(x) for x in sys.argv[1:7]); out.mkdir(parents=True, exist_ok=True)
NEWMETA = json.load(open(sys.argv[7], encoding='utf-8'))   # 2026-11-22: 새 학맥으로 다시 잡은 표본(교수·간선)만 쓴다
R, NPERM = 20, 1000; rng = np.random.default_rng(42); t0 = time.time()
def log(*x): print(f'[{time.time() - t0:5.0f}s]', *x, flush=True)
nz = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
sm = json.load(open(big / 'sample.json', encoding='utf-8')); nK, nS, nT = sm['nK'], sm['nS'], sm['nT']
seg_th = np.array(sm['seg_thesis']); seg_n = np.array(sm['seg_n'], dtype=np.float32); ktyp = np.array(sm['kci_typ'])
ids_t = json.load(open(big / 'seg_meta.json', encoding='utf-8'))['thesis_ids']
riss2doc = {}
for f in glob.glob(str(index / '*.json')):
    m = re.search(r'control_no=([0-9a-f]+)', open(f, encoding='utf-8').read(4000))
    if m: riss2doc[m.group(1)] = Path(f).stem
doc2riss = {v: k for k, v in riss2doc.items()}; by_ctrl = {}
for t in json.load(open(repo / 'site/data/research-public.json', encoding='utf-8'))['theses']:
    by_ctrl[t['id']] = t
    m = re.search(r'control_no=([0-9a-f]+)', t.get('riss_url') or '')
    if m: by_ctrl[m.group(1)] = t
meta = [by_ctrl.get(doc2riss.get(d, '')) or {} for d in ids_t]
gl = json.load(open(gen / 'site/data/genealogy-library.json', encoding='utf-8')); nodes = {x['id']: x for x in gl['nodes']}
profs = {i for i, x in nodes.items() if x.get('kind') == 'professor'}; advisor_of = {}; th_adv = collections.defaultdict(list)
for e in gl['edges']:
    if e.get('target') in profs and e.get('source') in profs and e['target'] != e['source']: advisor_of.setdefault(e['target'], e['source'])
    if e.get('thesis_id'): th_adv[e['thesis_id']].append(e['source'])
def root(p):
    seen = set(); cur = p
    while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
    return cur
uni = np.array([m.get('university') or '' for m in meta]); adv = np.array([(th_adv.get(doc2riss.get(d, '')) or [''])[0] for d in ids_t]); rt = np.array([root(a) if a else '' for a in adv])
def vectors(emb):
    Kv = nz(np.load(big / f'kci_{emb}.npy').astype(np.float32))
    for t in (0, 1): Kv[ktyp == t] -= Kv[ktyp == t].mean(axis=0)
    Kv = nz(Kv); Sv = nz(np.load(big / f'seg_{emb}.npy').astype(np.float32)); Sv = nz(Sv - Sv.mean(axis=0))
    T = np.zeros((nT, Sv.shape[1]), dtype=np.float32); np.add.at(T, seg_th, Sv * seg_n[:, None]); T = nz(T); T = nz(T - T.mean(axis=0))
    dr = np.array(sm['doc_rows']); dp = np.array(sm['doc_prof']); P = np.zeros((len(sm['pids']), Kv.shape[1]), dtype=np.float32); np.add.at(P, dp, Kv[dr]); P = nz(P)[PSEL]; P = nz(P - P.mean(axis=0))
    return T, P
def hdb(Xv, seed, nn, mcs):
    from umap import UMAP
    from hdbscan import HDBSCAN
    U = UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric='cosine', random_state=seed).fit_transform(Xv)
    return HDBSCAN(min_cluster_size=mcs, min_samples=max(2, mcs // 2), metric='euclidean', cluster_selection_method='eom').fit(U).labels_.astype(np.int32)
def km(Xv, seed, k): return KMeans(k, n_init=1, random_state=seed).fit(Xv).labels_.astype(np.int32)
def recur(L, others):
    k = L.max() + 1; rec = np.zeros(k)
    for O in others:
        for c in range(k):
            m = L == c; best = 0.0
            for o in set(O[m].tolist()) - {-1}:
                mo = O == o; best = max(best, (m & mo).sum() / (m | mo).sum())
            rec[c] += best >= 0.5
    return rec / len(others)
def top_n(lab, idx):
    v = lab[idx]; v = v[v != '']
    if len(v) == 0: return '', 0, 0
    a, n = collections.Counter(v.tolist()).most_common(1)[0]; return a, n, len(v)
def perm_p(lab, idx, obs_n):
    known = np.where(lab != '')[0]; m = int((lab[idx] != '').sum())
    if m < 3: return 1.0
    codes = np.unique(lab[known], return_inverse=True)[1]; hit = 0
    for _ in range(NPERM): hit += np.bincount(rng.choice(codes, size=m, replace=False)).max() >= obs_n
    return (hit + 1) / (NPERM + 1)
def bh(ps):
    ps = np.array(ps); o = np.argsort(ps); q = np.empty(len(ps)); prev = 1.0
    for rank, i in enumerate(o[::-1]):
        r_ = len(ps) - rank; prev = min(prev, ps[i] * len(ps) / r_); q[i] = prev
    return q
def pair_rate(labs, grp):
    num = den = 0
    for c in np.unique(grp):
        v = labs[grp == c]; v = v[v != '']; n = len(v)
        if n < 2: continue
        cnt = np.array(list(collections.Counter(v.tolist()).values())); num += (cnt * (cnt - 1) // 2).sum(); den += n * (n - 1) // 2
    return num / max(den, 1)
def cross_rate(grp, st_idx):
    a = b = den = 0
    for c in np.unique(grp):
        i = st_idx[grp == c]; i = i[(uni[i] != '') & (rt[i] != '')]
        if len(i) < 2: continue
        U = uni[i][:, None] == uni[i][None, :]; Rr = rt[i][:, None] == rt[i][None, :]; iu = np.triu_indices(len(i), 1)
        a += int((U[iu] & ~Rr[iu]).sum()); b += int((~U[iu] & Rr[iu]).sum()); den += len(iu[0])
    return a / max(den, 1), b / max(den, 1)
def analyze(runs):
    A = np.array([[ARI(a, b) for b in runs] for a in runs]); rep = int(((A.sum(axis=1) - 1) / (len(runs) - 1)).argmax()); L = runs[rep]
    rec = recur(L, [runs[i] for i in range(len(runs)) if i != rep]); st = np.where(rec >= 0.8)[0]
    rows = []
    for c in st:
        idx = np.where(L == c)[0]; u, un, um = top_n(uni, idx); r_, rn, rm = top_n(rt, idx); a_, an, am = top_n(adv, idx)
        o = idx[rt[idx] != r_] if r_ else idx; u2, un2, um2 = top_n(uni, o)
        rows.append({'n': len(idx), 'n_univ': len(set(uni[idx]) - {''}), 'n_adv': len(set(adv[idx]) - {''}), 'p_univ': perm_p(uni, idx, un), 'p_root': perm_p(rt, idx, rn), 'p_adv': perm_p(adv, idx, an), 'p_uo': perm_p(uni, o, un2) if um2 >= 3 else 1.0})
    if not rows: return {'stable': 0}
    for key in ('p_univ', 'p_root', 'p_adv', 'p_uo'):
        for r, v in zip(rows, bh([r[key] for r in rows])): r['q' + key[1:]] = float(v)
    kinds = collections.Counter(); km_ = collections.Counter()
    for r in rows:
        lin = r['q_root'] < 0.05 or r['q_adv'] < 0.05; uo = r['q_uo'] < 0.05; un_ = r['q_univ'] < 0.05
        kd = 'advisor' if lin and not uo else ('both' if lin and uo else ('univ' if (un_ or uo) else 'cross')); kinds[kd] += 1; km_[kd] += r['n']
    st_idx = np.where(np.isin(L, st))[0]; g = L[st_idx]; res = {}
    for nm, lab in (('same_advisor', adv), ('same_lineage', rt), ('same_univ', uni)):
        obs = pair_rate(lab[st_idx], g); nul = [pair_rate(lab[st_idx][rng.permutation(len(st_idx))], g) for _ in range(300)]
        res[nm] = {'obs': float(obs), 'null': float(np.mean(nul)), 'ratio': float(obs / max(np.mean(nul), 1e-12)), 'p': float((np.sum(np.array(nul) >= obs) + 1) / 301)}
    oa, ob = cross_rate(g, st_idx); nul = np.array([cross_rate(g[rng.permutation(len(g))], st_idx) for _ in range(300)])
    res['same_univ_diff_lineage'] = {'obs': float(oa), 'null': float(nul[:, 0].mean()), 'ratio': float(oa / max(nul[:, 0].mean(), 1e-12)), 'p': float((np.sum(nul[:, 0] >= oa) + 1) / 301)}
    res['diff_univ_same_lineage'] = {'obs': float(ob), 'null': float(nul[:, 1].mean()), 'ratio': float(ob / max(nul[:, 1].mean(), 1e-12)), 'p': float((np.sum(nul[:, 1] >= ob) + 1) / 301)}
    # 실행 20회 전부에서(되풀이 여부와 무관하게) 같은 묶음에 든 쌍의 비율 ÷ 묶인 사람 전체에서의 비율
    allr = collections.defaultdict(list)
    for Lr in runs:
        S = np.where(Lr >= 0)[0]; gS = Lr[S]; one = np.zeros(len(S), dtype=int)
        for nm, lab in (('same_advisor', adv), ('same_lineage', rt), ('same_univ', uni)): allr[nm].append(pair_rate(lab[S], gS) / max(pair_rate(lab[S], one), 1e-12))
        (xa, xb), (ya, yb) = cross_rate(gS, S), cross_rate(one, S); allr['same_univ_diff_lineage'].append(xa / max(ya, 1e-12)); allr['diff_univ_same_lineage'].append(xb / max(yb, 1e-12))
    pall = {k: {'ratio_mean': float(np.mean(v)), 'ratio_min': float(np.min(v)), 'ratio_max': float(np.max(v))} for k, v in allr.items()}
    clustered = float(np.mean([(Lr >= 0).mean() for Lr in runs])); ncl = float(np.mean([Lr.max() + 1 for Lr in runs]))
    return {'stable': int(len(st)), 'members': int(len(st_idx)), 'ari': float(A[np.triu_indices(len(runs), 1)].mean()), 'kinds': dict(kinds), 'kind_members': dict(km_),
            'median_univ': float(np.median([r['n_univ'] for r in rows])), 'median_adv': float(np.median([r['n_adv'] for r in rows])), 'median_n': float(np.median([r['n'] for r in rows])), 'pairs': res, 'pairs_all_runs': pall, 'clustered_share_mean': clustered, 'n_clusters_mean': ncl}
# ---- (나)용 관계 표지
PSEL = [i for i, p in enumerate(sm['pids']) if p in NEWMETA['professors']]; pids = [sm['pids'][i] for i in PSEL]; pidx = {p: i for i, p in enumerate(pids)}
_ke = {e['edge_id'] for e in NEWMETA['edges']}; ESEL = [i for i, e in enumerate(sm['edges']) if e['edge_id'] in _ke]; eidx = {sm['edges'][i]['edge_id']: k for k, i in enumerate(ESEL)}; stu_rows = np.array(sm['stu_rows'])[ESEL]
log('표본 교수', len(sm['pids']), '→', len(pids), '간선', len(sm['edges']), '→', len(ESEL))
RELS = ['advisor', 'sibling', 'uncle', 'grand', 'lineage_other', 'same_uni_outside_lineage', 'base']; rel = np.full((len(eidx), len(pids)), -1, dtype=np.int8)
# 2026-11-22: 관계는 공개 파생표 대신 새 학맥에서 바로 계산한다(stat_review.py 와 같은 규칙; 전체 간선)
students_of = collections.defaultdict(set)
for _s, _t in advisor_of.items(): students_of[_t].add(_s)
_roots = {p: root(p) for p in set(pids) | set(advisor_of) | set(advisor_of.values())}; _rsize = collections.Counter(_roots[p] for p in pids)
_punis = [set(NEWMETA['professors'][p]['universities']) for p in pids]; _emeta = {e['edge_id']: e for e in NEWMETA['edges']}
for eid, i in eidx.items():
    e = _emeta[eid]; a_ = e['professor_id']; u = e['university']; g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = (students_of.get(g, set()) - {a_}) if g else set(); r_ = _roots.get(a_)
    for j, pj in enumerate(pids):
        same = bool(u) and (u in _punis[j])
        if pj == a_: k = 0
        elif g and pj == g: k = 3
        elif pj in sibs: k = 1
        elif pj in uncles: k = 2
        elif _rsize.get(r_, 0) >= 2 and _roots.get(pj) == r_: k = 4
        elif same: k = 5
        else: k = 6
        rel[i, j] = k
assert (rel >= 0).all(), int((rel < 0).sum())
log('관계 쌍', {RELS[k]: int((rel == k).sum()) for k in range(len(RELS))})
def joint(T, P, nn=15, mcs=8):
    Xj = np.vstack([T, P]); runs = Parallel(n_jobs=5)(delayed(hdb)(Xj, s, nn, mcs) for s in range(R))
    CO = np.zeros(rel.shape, dtype=np.float64); pin = []; tin = []
    for Lr in runs:
        lt = Lr[:nT][stu_rows]; lp = Lr[nT:]; CO += (lt[:, None] == lp[None, :]) & (lt[:, None] >= 0); pin.append((lp >= 0).mean()); tin.append((lt >= 0).mean())
    CO /= len(runs); rate = {RELS[k]: float(CO[rel == k].mean()) for k in range(len(RELS))}
    boot = {k: [] for k in RELS[:-1]}
    for _ in range(500):
        cols = rng.integers(0, len(pids), len(pids)); Cb = CO[:, cols]; Rb = rel[:, cols]; base = Cb[Rb == 6].mean()
        for k in range(6):
            m = Rb == k
            if m.any(): boot[RELS[k]].append(Cb[m].mean() / max(base, 1e-12))
    return {'rate': rate, 'ratio': {k: rate[k] / max(rate['base'], 1e-12) for k in RELS[:-1]}, 'ratio_ci95': {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()},
            'n_pairs': {RELS[k]: int((rel == k).sum()) for k in range(len(RELS))}, 'prof_in_cluster': float(np.mean(pin)), 'thesis_in_cluster': float(np.mean(tin)), 'n_clusters_mean': float(np.mean([r.max() + 1 for r in runs]))}
res = {'thesis_only': {}, 'joint': {}}
CONF = [('HDBSCAN 이웃 15·최소 8 (기본)', 'hdb', 15, 8), ('HDBSCAN 이웃 15·최소 5', 'hdb', 15, 5), ('HDBSCAN 이웃 15·최소 12', 'hdb', 15, 12), ('HDBSCAN 이웃 15·최소 20', 'hdb', 15, 20), ('HDBSCAN 이웃 10·최소 8', 'hdb', 10, 8), ('HDBSCAN 이웃 30·최소 8', 'hdb', 30, 8), ('k-means 40개', 'km', 40, 0)]
for emb in ['ko_sroberta', 'bge_m3', 'e5_large']:
    T, P = vectors(emb); res['thesis_only'][emb] = {}
    for name, kind, a, b in CONF:
        runs = Parallel(n_jobs=5)(delayed(hdb)(T, s, a, b) for s in range(R)) if kind == 'hdb' else Parallel(n_jobs=5)(delayed(km)(T, s, a) for s in range(R))
        r = analyze(runs); res['thesis_only'][emb][name] = r
        log(emb, name, 'stable', r.get('stable'), 'members', r.get('members'), r.get('kinds'), {k: round(v['ratio'], 2) for k, v in r.get('pairs', {}).items()}, 'all', {k: round(v['ratio_mean'], 2) for k, v in r.get('pairs_all_runs', {}).items()})
    res['joint'][emb] = {'main': joint(T, P), 'mcs5': joint(T, P, 15, 5), 'mcs12': joint(T, P, 15, 12)}
    log(emb, 'joint', {k: (round(res['joint'][emb]['main']['rate'][k] * 100, 2), round(res['joint'][emb]['main']['ratio'].get(k, 1), 1)) for k in RELS}, 'prof in cluster', round(res['joint'][emb]['main']['prof_in_cluster'], 2))
    json.dump(res, open(out / 'reverse_refined.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('done')
