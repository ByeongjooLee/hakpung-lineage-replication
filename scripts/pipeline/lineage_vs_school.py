"""대학 효과 × 계보 효과 분리 (2026-10-29) — school_effect.py 와 같은 입력(vectors.npz, vectors_meta.json), 논문·교수 고정효과 잔차 기준.

계보 = 교수 사이의 박사 지도 관계(genealogy-library.json 간선 중 target 이 교수 노드인 것)를 따라 올라간 뿌리 교수. 같은 뿌리면 같은 계보.
논문 i(지도교수 a, 대학 u)에 대해 다른 교수 j 를 서로 배타적인 유형으로 나눈다:
  advisor      지도교수 본인
  grand        조부(지도교수의 지도교수)
  sibling      형제 교수(지도교수 a 의 다른 제자로 교수가 된 사람)
  uncle        삼촌(조부의 다른 제자 교수)
  lineage      그 밖에 같은 뿌리의 교수(증조부·사촌 등)
  none         계보 밖
그리고 각 유형을 같은 대학/다른 대학으로 다시 가른다(2×k 표). 회귀: 잔차 = Σβ·유형더미 + β_u·같은대학(계보 밖) + β_ul·같은대학×계보 안 ; 논문 클러스터 SE.
출력(--out): lineage_cells.csv, lineage_regression.json, lineage_roots.csv, 계보분리_보고서.md
"""
import argparse, json, csv, collections, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--min-cell', type=int, default=30)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P0 = Path(a.prelim); z = np.load(P0 / 'vectors.npz', allow_pickle=True); meta = json.load(open(P0 / 'vectors_meta.json', encoding='utf-8'))
    STU, PROF, pids = z['stu'], z['prof'], list(z['pids']); edges = meta['edges']; pinfo = meta['professors']
    S_raw = STU @ PROF.T; S = S_raw - S_raw.mean(axis=1, keepdims=True) - S_raw.mean(axis=0, keepdims=True) + S_raw.mean()
    pidx = {p: i for i, p in enumerate(pids)}; n = len(pids)
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    advisor_of = {}  # 교수 → 지도교수(교수 노드) ; 묶음에 없는 교수도 포함해 뿌리를 찾는다
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']:
            advisor_of.setdefault(e['target'], e['source'])
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen:
            seen.add(cur); cur = advisor_of[cur]
        return cur
    students_of = collections.defaultdict(set)
    for s, t in advisor_of.items(): students_of[t].add(s)
    roots = {p: root(p) for p in pids}
    root_size = collections.Counter(roots.values())
    prof_unis = [set(pinfo[p]['universities']) for p in pids]
    uni_e = [e['university'] for e in edges]

    TYPES = ['advisor', 'grand', 'sibling', 'uncle', 'lineage', 'none']
    cells = collections.defaultdict(list)  # (type, same_uni) -> sims
    X_rows = []; y_rows = []; cl = []
    per_root = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, e in enumerate(edges):
        a_ = e['professor_id']; ai = pidx[a_]; u = uni_e[i]
        g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = students_of.get(g, set()) - {a_} if g else set()
        r = roots.get(a_, root(a_))
        for j, pj in enumerate(pids):
            if pj == a_: t = 'advisor'
            elif g and pj == g: t = 'grand'
            elif pj in sibs: t = 'sibling'
            elif pj in uncles: t = 'uncle'
            elif root_size.get(r, 0) >= 2 and roots.get(pj) == r: t = 'lineage'
            else: t = 'none'
            same = bool(u) and (u in prof_unis[j])
            cells[(t, same)].append(S[i, j])
            if t != 'advisor': per_root[r][(t, same)].append(S[i, j])
            d = [1.0 if t == k else 0.0 for k in TYPES[:-1]]  # advisor, grand, sibling, uncle, lineage
            d += [1.0 if (same and t == 'none') else 0.0, 1.0 if (same and t in ('grand', 'sibling', 'uncle', 'lineage')) else 0.0]
            X_rows.append(d); y_rows.append(S[i, j]); cl.append(i)
    X = np.array(X_rows); y = np.array(y_rows); cl = np.array(cl)
    names = ['advisor', 'grand', 'sibling', 'uncle', 'lineage_other', 'same_uni_outside_lineage', 'same_uni_x_lineage']
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1])); order = np.argsort(cl, kind='stable'); Xs, rs, cs = X[order], resid[order], cl[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1; starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(cs)]])
    for s0, e0 in zip(starts, ends):
        sgrad = Xs[s0:e0].T @ rs[s0:e0]; meat += np.outer(sgrad, sgrad)
    V = XtX_inv @ meat @ XtX_inv; se = np.sqrt(np.diag(V))
    reg = {k: {'beta': float(b), 'se': float(s), 't': float(b / s)} for k, b, s in zip(names, beta, se)}
    reg['_note'] = '종속변수 = 코사인 − 논문 평균 − 교수 평균 + 전체 평균. 기준 범주 = 다른 대학·계보 밖 교수. 논문 단위 클러스터 SE.'
    json.dump(reg, open(out / 'lineage_regression.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    L = ['# 대학 효과 × 계보 효과 분리', '', f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 논문 {len(edges)}편 × 교수 {n}명, 잔차 코사인(논문·교수 고정효과 제거). 계보 뿌리 {len(set(roots.values()))}개, 2명 이상 계보 {sum(1 for v in root_size.values() if v >= 2)}개.', '',
         '## 1. 관계 유형 × 같은 대학 여부별 평균 잔차 유사도', '| 유형 | 같은 대학: 평균 (n) | 다른 대학: 평균 (n) | 차이(같은−다른) |', '|---|---|---|---|']
    rows_c = []
    for t in TYPES:
        s_ = cells.get((t, True), []); d_ = cells.get((t, False), [])
        ms = np.mean(s_) if len(s_) >= a.min_cell else float('nan'); md = np.mean(d_) if len(d_) >= a.min_cell else float('nan')
        L.append(f'| {t} | {ms:+.4f} ({len(s_)}) | {md:+.4f} ({len(d_)}) | {(ms - md):+.4f} |')
        rows_c.append({'type': t, 'same_uni_mean': ms, 'same_uni_n': len(s_), 'diff_uni_mean': md, 'diff_uni_n': len(d_)})
    with open(out / 'lineage_cells.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_c[0].keys())); w.writeheader(); w.writerows(rows_c)
    L += ['', '## 2. 회귀 (기준: 다른 대학·계보 밖)', '| 항 | 계수 | 클러스터 SE | t |', '|---|---|---|---|']
    label = {'advisor': '지도교수', 'grand': '조부(지도교수의 지도교수)', 'sibling': '형제 교수(같은 지도교수)', 'uncle': '삼촌(조부의 다른 제자)', 'lineage_other': '같은 계보 기타', 'same_uni_outside_lineage': '같은 대학, 계보 밖', 'same_uni_x_lineage': '같은 대학 × 계보 안(추가분)'}
    for k in names: L.append(f"| {label[k]} | {reg[k]['beta']:+.4f} | {reg[k]['se']:.4f} | {reg[k]['t']:.1f} |")
    # 계보별
    L += ['', '## 3. 뿌리 교수별 (논문 100쌍 이상인 계보)', '| 뿌리 교수 | 계보 교수 수 | 조부(같은 대학/다른) | 형제·삼촌·기타 계보(같은/다른) | 같은 대학·계보 밖 | 다른 대학·계보 밖 |', '|---|---|---|---|---|---|']
    rows_r = []
    def m(xs): return f'{np.mean(xs):+.4f}' if len(xs) >= 10 else '—'
    for r, cd in sorted(per_root.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        tot = sum(len(v) for v in cd.values())
        if tot < 100 or root_size.get(r, 0) < 2: continue
        lin_s = cd[('sibling', True)] + cd[('uncle', True)] + cd[('lineage', True)]; lin_d = cd[('sibling', False)] + cd[('uncle', False)] + cd[('lineage', False)]
        nm_ = profs_all.get(r, {}).get('name', r)
        L.append(f"| {nm_} | {root_size[r]} | {m(cd[('grand', True)])} / {m(cd[('grand', False)])} | {m(lin_s)} / {m(lin_d)} | {m(cd[('none', True)])} | {m(cd[('none', False)])} |")
        rows_r.append({'root': nm_, 'root_id': r, 'lineage_professors': root_size[r], 'grand_same_uni': np.mean(cd[('grand', True)]) if cd[('grand', True)] else '', 'grand_diff_uni': np.mean(cd[('grand', False)]) if cd[('grand', False)] else '',
                       'lineage_same_uni': np.mean(lin_s) if lin_s else '', 'lineage_diff_uni': np.mean(lin_d) if lin_d else '', 'none_same_uni': np.mean(cd[('none', True)]) if cd[('none', True)] else '', 'none_diff_uni': np.mean(cd[('none', False)]) if cd[('none', False)] else ''})
    with open(out / 'lineage_roots.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_r[0].keys())); w.writeheader(); w.writerows(rows_r)
    # 대학별: 같은 대학 교수 가운데 계보 안 비율
    L += ['', '## 4. 대학별로 "같은 대학 교수"가 계보 안인 비율 (논문 기준)', '| 대학 | 논문 | 같은 대학 교수 쌍 | 그중 계보 안 | 계보 안 비율 |', '|---|---|---|---|---|']
    by_u = collections.defaultdict(lambda: [0, 0, 0])
    for i, e in enumerate(edges):
        u = uni_e[i]
        if not u: continue
        a_ = e['professor_id']; r = roots.get(a_, root(a_)); g = advisor_of.get(a_)
        by_u[u][0] += 1
        for j, pj in enumerate(pids):
            if pj == a_ or u not in prof_unis[j]: continue
            by_u[u][1] += 1
            if (g and pj == g) or pj in students_of.get(a_, set()) or (root_size.get(r, 0) >= 2 and roots[pj] == r): by_u[u][2] += 1
    for u, (nt, ns, nl) in sorted(by_u.items(), key=lambda kv: -kv[1][0])[:15]:
        L.append(f'| {u} | {nt} | {ns} | {nl} | {nl / ns if ns else 0:.2f} |')
    L += ['', '## 읽는 법', '- 1·2 에서 "같은 대학, 계보 밖" 이 0 에 가깝고 조부·형제·삼촌이 양이면, 앞서 본 대학 효과는 사실 계보 효과였다는 뜻. 반대로 계보 밖에서도 양이면 계보와 별개의 학풍이 있다.',
          '- "같은 대학 × 계보 안(추가분)" 은 계보 안 교수가 같은 대학에 있을 때 더 닮는지(학교가 계보의 전달을 돕는지)를 본다.', '- 3 은 계보별 사례. 4 는 서울대처럼 같은 대학 교수가 곧 계보인 곳에서 둘이 분리되지 않음을 보인다.']
    (out / '계보분리_보고서.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({k: round(v['t'], 1) if isinstance(v, dict) else v for k, v in reg.items() if k != '_note'}, ensure_ascii=False))
    print(json.dumps({k: round(v['beta'], 4) for k, v in reg.items() if k != '_note'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
