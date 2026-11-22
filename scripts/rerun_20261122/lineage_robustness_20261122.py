"""계보 강건성 분석 (2026-11-01, 연구자 「돌려」 — 서울대 계보 지배에 대한 심사 대응)

입력: export_derived_tables.py 가 만든 공개 파생표 폴더(theses.csv, professors.csv, pair_relations_similarity.csv.gz)만 쓴다
      — 파생표만으로 재계산이 되는지 함께 확인하는 뜻. 뿌리 교수의 이름·소속은 저장소 genealogy-library.json 에서 읽는다.
세 분석(표현마다, 관계 더미 7개 OLS, 종속변수 = 그 표본 행렬에서 다시 잔차화한 코사인):
  A. 삼방향 클러스터 SE — 논문·교수·계보 뿌리(논문 지도교수의 뿌리). Cameron–Gelbach–Miller 포함-배제 공식. 논문은 뿌리에 중첩되므로
     V = V_교수 + V_뿌리 − V_(교수×뿌리) 로 줄어든다(공식을 그대로 계산해 확인). 뿌리 클러스터 수가 적어 와일드 클러스터 부트스트랩(Rademacher, 뿌리 단위) p 도 보고.
  B. 계보 하나씩 빼기 — 정한모 · 전광용 · 정한모+전광용 · 최동호 · 서울대 뿌리 전부(정한모·전광용·신범순·박성창·조동일) 를 뺀 표본에서 다시 잔차화·추정(양방향 클러스터 SE).
  C. 층화 — 서울대 뿌리 계보 논문 / 그 밖 논문을 따로 추정하고, 전체 행렬에서 관계×서울대 상호작용 모형으로 차이를 검정(양방향 클러스터 SE).
출력(--out): 계보강건성_보고서.md, lineage_robustness.json, lineage_robustness_results.csv
실행: py -X utf8 lineage_robustness.py --tables <공개_파생표_YYYYMMDD> --repo <<SITE>> --out <폴더> [--reps bge,bge_subset,tfidf,lsa300,lda50] [--wild 499] [--seed 42]
"""
import argparse, json, csv, collections, time
from pathlib import Path
import numpy as np
import pandas as pd

REL = ['advisor', 'grand', 'sibling', 'uncle', 'lineage_other', 'same_uni_outside_lineage', 'same_uni_x_lineage']
LAB = {'advisor': '지도교수', 'grand': '조부 교수', 'sibling': '형제 교수', 'uncle': '삼촌 교수', 'lineage_other': '같은 계보 기타', 'same_uni_outside_lineage': '계보 밖 같은 대학', 'same_uni_x_lineage': '같은 대학×계보'}
SHOW = ['advisor', 'sibling', 'uncle', 'grand', 'same_uni_outside_lineage']


def resid2(S):
    return S - S.mean(axis=1, keepdims=True) - S.mean(axis=0, keepdims=True) + S.mean()


def pnorm2(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def cluster_V(X, e, cl, XtX_inv):
    """클러스터 샌드위치. cl 은 0..G-1 정수 배열."""
    G = np.zeros((cl.max() + 1, X.shape[1])); np.add.at(G, cl, X * e[:, None]); return XtX_inv @ (G.T @ G) @ XtX_inv


def codes(a):
    _, inv = np.unique(a, return_inverse=True); return inv


def fit(X, y, clusters):
    """OLS + 지정 클러스터 조합의 SE. clusters = {이름: 정수 배열}. 반환 beta, {se_이름}, 2way(thesis,prof), 3way(thesis,prof,root)."""
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; e = y - X @ beta
    V = {k: cluster_V(X, e, c, XtX_inv) for k, c in clusters.items()}
    V0 = XtX_inv @ ((X * e[:, None]).T @ (X * e[:, None])) @ XtX_inv
    out = {'beta': beta, 'e': e, 'XtX_inv': XtX_inv, 'V0': V0, 'V': V}
    if {'thesis', 'prof'} <= set(V): out['V2'] = V['thesis'] + V['prof'] - V0
    if {'thesis', 'prof', 'root', 'prof_root'} <= set(V):
        # CGM 3-way: 단일 3 − 쌍 3 + 삼중. 논문∩교수 = 관측, 논문∩뿌리 = 논문(중첩), 교수∩뿌리 = prof_root, 삼중 = 관측
        out['V3'] = V['thesis'] + V['prof'] + V['root'] - V0 - V['thesis'] - V['prof_root'] + V0
        out['V3_reduced'] = V['prof'] + V['root'] - V['prof_root']
    return out


def se_of(V): return np.sqrt(np.maximum(np.diag(V), 1e-18))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tables', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--reps', default='bge,bge_subset,tfidf,lsa300,lda50'); ap.add_argument('--wild', type=int, default=499); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(); T = Path(a.tables); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:5.0f}s]', *x, flush=True)

    th = pd.read_csv(T / 'theses.csv', dtype=str, keep_default_na=False); pr = pd.read_csv(T / 'professors.csv', dtype=str, keep_default_na=False)
    pairs = pd.read_csv(T / 'pair_relations_similarity.csv.gz', dtype={'edge_id': str, 'professor_id': str, 'relation_all': str}, keep_default_na=False)
    log('pairs', len(pairs))
    edge_ids = list(th['edge_id']); pids = list(pr['professor_id']); ei = {e: i for i, e in enumerate(edge_ids)}; pj = {p: j for j, p in enumerate(pids)}
    n_e, n_p = len(edge_ids), len(pids)
    I = pairs['edge_id'].map(ei).to_numpy(); J = pairs['professor_id'].map(pj).to_numpy()
    assert len(pairs) == n_e * n_p and not np.isnan(I.astype(float)).any()
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8')); nodes = {x['id']: x for x in gl['nodes']}
    root_of_edge = th['lineage_root_all'].to_numpy()
    def rname(r): return nodes.get(r, {}).get('name', r)
    snu_roots = sorted({r for r in set(root_of_edge) if '서울대' in (nodes.get(r, {}).get('affiliations') or [])})
    log('서울대 뿌리', [(rname(r), int((root_of_edge == r).sum())) for r in snu_roots])
    # 관계 설계행렬 (전체 간선 단계)
    Xfull = np.zeros((n_e, n_p, len(REL)))
    rel = pairs['relation_all'].to_numpy(); sx = pairs['same_uni_x_lineage_all'].to_numpy().astype(int)
    for k, n in enumerate(REL[:6]): Xfull[I[rel == n], J[rel == n], k] = 1
    Xfull[I[sx == 1], J[sx == 1], 6] = 1
    def root_id(name):
        c = [r for r in set(root_of_edge) if rname(r) == name]; assert len(c) == 1, (name, c); return c[0]
    drops = [('정한모 계보 제외', [root_id('정한모')]), ('전광용 계보 제외', [root_id('전광용')]), ('정한모+전광용 제외', [root_id('정한모'), root_id('전광용')]),
             ('정한숙 계보 제외', [root_id('정한숙')]), ('서울대 뿌리 계보 전부 제외', snu_roots)]
    is_snu = np.isin(root_of_edge, snu_roots)

    results = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'tables': T.name, 'n_professors': n_p, 'snu_roots': [rname(r) for r in snu_roots], 'wild_reps': a.wild, 'reps': {}}
    rows_out = []
    for rep in a.reps.split(','):
        raw = pairs[f'raw_{rep}'].to_numpy()
        have = raw != ''
        rows_have = np.unique(I[have]); assert have.sum() == len(rows_have) * n_p, rep
        S = np.full((n_e, n_p), np.nan); S[I[have], J[have]] = raw[have].astype(float); S = S[rows_have]
        X_r = Xfull[rows_have]; roots_r = root_of_edge[rows_have]; snu_r = is_snu[rows_have]; m = len(rows_have)
        log(rep, '논문', m)
        R = {'n_theses': int(m)}
        def design(Xm):
            X = Xm.reshape(-1, len(REL)); keep = X.sum(axis=0) > 0; return X[:, keep], [n for n, k in zip(REL, keep) if k]
        def run(rowsel, label, want3=False):
            Ss = resid2(S[rowsel]); X, names = design(X_r[rowsel]); y = Ss.reshape(-1); mm = int(rowsel.sum())
            cl = {'thesis': np.repeat(np.arange(mm), n_p), 'prof': np.tile(np.arange(n_p), mm)}
            if want3:
                rc = codes(roots_r[rowsel]); cl['root'] = np.repeat(rc, n_p); cl['prof_root'] = codes(cl['root'] * n_p + cl['prof'])
            f = fit(X, y, cl); se2 = se_of(f['V2']); sd = float(y.std())
            res = {'label': label, 'n_theses': mm, 'sd_resid': sd, 'terms': {}}
            for k, n in enumerate(names):
                res['terms'][n] = {'beta': float(f['beta'][k]), 'se_twoway': float(se2[k]), 't_twoway': float(f['beta'][k] / se2[k]), 'std_effect': float(f['beta'][k] / sd), 'n_pairs': int(X[:, k].sum())}
            return f, X, y, names, cl, res
        # ---- A. 삼방향 클러스터 + 와일드 부트스트랩(뿌리)
        allrows = np.ones(m, dtype=bool)
        f, X, y, names, cl, resA = run(allrows, '전체', want3=True)
        se3 = se_of(f['V3']); se3r = se_of(f['V3_reduced']); se_root = se_of(f['V']['root']); n_root = int(cl['root'].max() + 1)
        assert np.allclose(f['V3'], f['V3_reduced'])
        # 와일드 클러스터 부트스트랩: 뿌리 단위 Rademacher, 비제약(unrestricted) — t* = (b*−b)/se3*
        Xb = X @ f['beta']; e = f['e']; XtX_inv = f['XtX_inv']; tstat = f['beta'] / se3; cnt = np.zeros(len(names)); rc = cl['root']
        for _ in range(a.wild):
            w = rng.choice([-1.0, 1.0], size=n_root)[rc]; ys = Xb + e * w
            b = XtX_inv @ X.T @ ys; es = ys - X @ b
            V0 = XtX_inv @ ((X * es[:, None]).T @ (X * es[:, None])) @ XtX_inv
            Vs = cluster_V(X, es, cl['prof'], XtX_inv) + cluster_V(X, es, rc, XtX_inv) - cluster_V(X, es, cl['prof_root'], XtX_inv)
            ts = (b - f['beta']) / se_of(Vs); cnt += (np.abs(ts) >= np.abs(tstat))
        for k, n in enumerate(names):
            resA['terms'][n].update({'se_threeway': float(se3[k]), 't_threeway': float(f['beta'][k] / se3[k]), 'p_threeway': pnorm2(f['beta'][k] / se3[k]), 'se_root_only': float(se_root[k]), 'wild_p': float(cnt[k] / a.wild)})
        resA['n_root_clusters'] = n_root; R['A_threeway'] = resA
        log(rep, 'A', {n: (round(resA['terms'][n]['beta'], 4), round(resA['terms'][n]['t_twoway'], 1), round(resA['terms'][n]['t_threeway'], 1), resA['terms'][n]['wild_p']) for n in SHOW if n in resA['terms']})
        # ---- B. 계보 하나씩 빼기
        R['B_leave_out'] = []
        for label, rs in drops:
            sel = ~np.isin(roots_r, rs)
            _, _, _, _, _, res = run(sel, label); res['dropped_theses'] = int((~sel).sum()); R['B_leave_out'].append(res)
            log(rep, 'B', label, res['n_theses'], {n: (round(res['terms'][n]['beta'], 4), round(res['terms'][n]['t_twoway'], 1), res['terms'][n]['n_pairs']) for n in SHOW if n in res['terms']})
        # ---- C. 층화 + 상호작용
        _, _, _, _, _, resS = run(snu_r, '서울대 뿌리 계보만'); _, _, _, _, _, resN = run(~snu_r, '서울대 뿌리 계보 밖')
        Ss = resid2(S); X, names = design(X_r); y = Ss.reshape(-1); g = np.repeat(snu_r.astype(float), n_p)
        Xi = np.hstack([X, X * g[:, None]]); keep = Xi.sum(axis=0) > 0; Xi = Xi[:, keep]; inames = [n for n in names] + [n + '×서울대' for n in names]; inames = [n for n, k in zip(inames, keep) if k]
        fi = fit(Xi, y, {'thesis': np.repeat(np.arange(m), n_p), 'prof': np.tile(np.arange(n_p), m)}); sei = se_of(fi['V2'])
        inter = {}
        for k, n in enumerate(inames):
            if n.endswith('×서울대'): inter[n[:-4]] = {'diff': float(fi['beta'][k]), 'se_twoway': float(sei[k]), 't': float(fi['beta'][k] / sei[k]), 'p': pnorm2(fi['beta'][k] / sei[k])}
        R['C_strata'] = {'snu': resS, 'non_snu': resN, 'interaction': inter}
        log(rep, 'C', {n: (round(resS['terms'].get(n, {}).get('beta', float('nan')), 4), round(resN['terms'].get(n, {}).get('beta', float('nan')), 4), round(inter.get(n, {}).get('t', float('nan')), 1)) for n in SHOW})
        results['reps'][rep] = R
        for n in names:
            t_ = resA['terms'][n]
            rows_out.append({'representation': rep, 'analysis': 'A_threeway', 'subset': '전체', 'relation': n, 'beta': t_['beta'], 'se_twoway': t_['se_twoway'], 't_twoway': t_['t_twoway'], 'se_threeway': t_['se_threeway'], 't_threeway': t_['t_threeway'], 'p_threeway': t_['p_threeway'], 'wild_p_root': t_['wild_p'], 'n_pairs': t_['n_pairs'], 'n_theses': m, 'n_root_clusters': n_root})
        for res in R['B_leave_out'] + [resS, resN]:
            for n, t_ in res['terms'].items():
                rows_out.append({'representation': rep, 'analysis': 'B_leave_out' if res in R['B_leave_out'] else 'C_strata', 'subset': res['label'], 'relation': n, 'beta': t_['beta'], 'se_twoway': t_['se_twoway'], 't_twoway': t_['t_twoway'], 'n_pairs': t_['n_pairs'], 'n_theses': res['n_theses']})
        for n, d in inter.items():
            rows_out.append({'representation': rep, 'analysis': 'C_interaction', 'subset': '서울대 − 그 밖 (차이)', 'relation': n, 'beta': d['diff'], 'se_twoway': d['se_twoway'], 't_twoway': d['t'], 'p_threeway': d['p'], 'n_theses': m})
    json.dump(results, open(out / 'lineage_robustness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    cols = ['representation', 'analysis', 'subset', 'relation', 'beta', 'se_twoway', 't_twoway', 'se_threeway', 't_threeway', 'p_threeway', 'wild_p_root', 'n_pairs', 'n_theses', 'n_root_clusters']
    with open(out / 'lineage_robustness_results.csv', 'w', encoding='utf-8-sig', newline='') as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=cols); w.writeheader()
        for r in rows_out: w.writerow({c: (f'{r[c]:.6f}' if isinstance(r.get(c), float) else r.get(c, '')) for c in cols})

    # ---- 보고서
    def fmt(b, t): return f'{b:+.4f} ({t:.1f})'
    L = [f'# 계보 강건성 — 서울대 계보 지배에 대한 검토', '', f"생성 {results['generated']}. 입력 = 공개 파생표 `{T.name}`(theses·professors·pair_relations_similarity). 교수 {n_p}명. 서울대 뿌리 계보 = {', '.join(results['snu_roots'])}. 와일드 클러스터 부트스트랩 {a.wild}회(뿌리 단위 Rademacher).", '',
         '## A. 계보 뿌리를 더한 삼방향 클러스터 SE', '', '논문·교수 양방향에 「논문 지도교수의 계보 뿌리」 클러스터를 더했다. 논문이 뿌리에 중첩되므로 삼방향 분산은 V(교수)+V(뿌리)−V(교수×뿌리)와 같다(계산으로 확인). 뿌리 클러스터 수가 적어 와일드 부트스트랩 p 를 함께 본다.', '',
         '| 표현 | 항 | 계수 | t 양방향 | **t 삼방향** | p 삼방향 | 와일드 p(뿌리) | 쌍 | 뿌리 수 |', '|---|---|---|---|---|---|---|---|---|']
    for rep, R in results['reps'].items():
        A = R['A_threeway']
        for n in SHOW:
            if n not in A['terms']: continue
            t_ = A['terms'][n]; L.append(f"| {rep} | {LAB[n]} | {t_['beta']:+.4f} | {t_['t_twoway']:.1f} | **{t_['t_threeway']:.1f}** | {t_['p_threeway']:.2g} | {t_['wild_p']:.3f} | {t_['n_pairs']} | {A['n_root_clusters']} |")
    L += ['', '## B. 계보 하나씩 빼기 (양방향 클러스터 SE, 빼고 남은 표본에서 다시 잔차화)', '']
    for rep, R in results['reps'].items():
        L += [f'### {rep}', '', '| 표본 | 논문 | 지도교수 | 형제 | 삼촌 | 조부 (쌍) | 계보 밖 같은 대학 |', '|---|---|---|---|---|---|---|']
        for res in [R['A_threeway']] + R['B_leave_out']:
            def c(n):
                t_ = res['terms'].get(n); return '—' if not t_ else fmt(t_['beta'], t_['t_twoway']) + (f" {t_['n_pairs']}" if n == 'grand' else '')
            L.append(f"| {res['label']} | {res['n_theses']} | {c('advisor')} | {c('sibling')} | {c('uncle')} | {c('grand')} | {c('same_uni_outside_lineage')} |")
        L.append('')
    L += ['## C. 층화 — 서울대 뿌리 계보 안 / 밖, 그리고 차이 검정', '', '층별 추정은 각 층 행렬을 따로 잔차화한 것. 차이 검정은 전체 행렬의 관계×서울대 상호작용 모형(양방향 클러스터 SE).', '',
          '| 표현 | 항 | 서울대 계보 안 (쌍) | 그 밖 (쌍) | 차이 안−밖 | t 차이 | p |', '|---|---|---|---|---|---|---|']
    for rep, R in results['reps'].items():
        C = R['C_strata']
        for n in SHOW:
            s, o, d = C['snu']['terms'].get(n), C['non_snu']['terms'].get(n), C['interaction'].get(n)
            L.append(f"| {rep} | {LAB[n]} | {fmt(s['beta'], s['t_twoway']) + ' ' + str(s['n_pairs']) if s else '—'} | {fmt(o['beta'], o['t_twoway']) + ' ' + str(o['n_pairs']) if o else '—'} | {d['diff']:+.4f} | {d['t']:.1f} | {d['p']:.2g} |" if d else f"| {rep} | {LAB[n]} | — | — | — | — | — |")
    L += ['', f"서울대 계보 안 논문 {int(is_snu.sum())}편 / 밖 {int((~is_snu).sum())}편(bge 표본 기준).", '',
          '## 읽는 법', '', '- A 의 t 삼방향이 양방향과 비슷하면 계보 안 의존성이 추정 정밀도를 크게 깎지 않는다는 뜻. 와일드 p 는 뿌리 수가 적을 때의 보수적 p.',
          '- B 에서 어느 한 계보를 빼도 지도교수·형제·삼촌 계수가 비슷하면 결과를 한 계보가 끌고 가지 않는다. 조부 항은 쌍 수가 크게 줄어 구간이 넓어진다 — 비유의를 「효과 없음」으로 읽지 말 것.',
          '- C 의 차이 검정이 비유의면 서울대 계보와 그 밖에서 효과 크기가 다르다는 증거가 없다. 서울대 밖 층의 조부 쌍 수를 반드시 함께 적을 것.']
    (out / '계보강건성_보고서.md').write_text('\n'.join(L), encoding='utf-8'); log('report', out / '계보강건성_보고서.md')


if __name__ == '__main__':
    main()
