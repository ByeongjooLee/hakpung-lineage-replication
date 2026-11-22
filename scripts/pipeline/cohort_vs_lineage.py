"""형제 교수 효과 vs 코호트 효과 (2026-10-29) — lineage_vs_school.py 와 같은 입력, 잔차 코사인.

교수 j 의 박사학위 연도 y_j(학맥 간선에서 j 가 target 인 간선의 연도 → 없으면 교수 노드 thesis_ids 의 research-public 연도)와
논문 i 의 연도 y_i 의 차 gap = y_i − y_j 로:
  형제 교수(같은 지도교수)를 |gap| ≤ 5 / 6–15 / > 15 / 미상으로 나눔 — 코호트(같은 시기 수학)면 가까운 쪽만 크고, 지도교수를 통한 전승이면 멀어도 유지
  계보 밖 같은 대학 교수도 같은 구간 — 순수 코호트(다른 지도교수 아래 같은 시기 같은 대학원) 효과
  지도교수 본인은 지도교수 학위 후 경과 연수(y_i − y_a)로 ≤ 10 / 11–20 / 21–30 / > 30 / 미상 — 교수 경력 단계별 전승 강도
회귀: 잔차 = Σβ·더미 (기준 = 다른 대학·계보 밖), 논문 클러스터 SE. 형제 쌍만 따로 gap 연속값 회귀.
출력(--out): cohort_regression.json, sibling_gap.csv, 코호트분리_보고서.md
"""
import argparse, json, csv, collections, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--phd-years', help='fill_prof_phd_years.py 출력(prof_phd_years.csv) — 교수 학위 연도 보강')
    ap.add_argument('--suffix', default='', help='출력 파일 이름 꼬리(예: _보강)')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P0 = Path(a.prelim); z = np.load(P0 / 'vectors.npz', allow_pickle=True); meta = json.load(open(P0 / 'vectors_meta.json', encoding='utf-8'))
    STU, PROF, pids = z['stu'], z['prof'], list(z['pids']); edges = meta['edges']; pinfo = meta['professors']
    S_raw = STU @ PROF.T; S = S_raw - S_raw.mean(axis=1, keepdims=True) - S_raw.mean(axis=0, keepdims=True) + S_raw.mean()
    pidx = {p: i for i, p in enumerate(pids)}
    repo = Path(a.repo)
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    rp = json.load(open(repo / 'site/data/research-public.json', encoding='utf-8')); th_year = {t['id']: t.get('year') for t in rp['theses']}
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    advisor_of = {}; phd_year = {}
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']:
            advisor_of.setdefault(e['target'], e['source'])
            if str(e.get('year', '')).isdigit(): phd_year.setdefault(e['target'], int(e['year']))
    for p, node in profs_all.items():
        if p in phd_year: continue
        for tid in node.get('thesis_ids') or []:
            y = th_year.get(tid)
            if str(y).isdigit(): phd_year[p] = int(y); break
    if a.phd_years:
        for r in csv.DictReader(open(a.phd_years, encoding='utf-8-sig', newline='')):
            if str(r.get('phd_year', '')).isdigit() and r['professor_id'] not in phd_year: phd_year[r['professor_id']] = int(r['phd_year'])
    students_of = collections.defaultdict(set)
    for s, t in advisor_of.items(): students_of[t].add(s)
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
        return cur
    roots = {p: root(p) for p in pids}; root_size = collections.Counter(roots.values())
    prof_unis = [set(pinfo[p]['universities']) for p in pids]
    cov = sum(1 for p in pids if p in phd_year)
    GAPS = [('≤5', lambda g: abs(g) <= 5), ('6–15', lambda g: 5 < abs(g) <= 15), ('>15', lambda g: abs(g) > 15)]
    AGAPS = [('≤10', lambda g: g <= 10), ('11–20', lambda g: 10 < g <= 20), ('21–30', lambda g: 20 < g <= 30), ('>30', lambda g: g > 30)]
    names = ['advisor_' + k for k, _ in AGAPS] + ['advisor_unknown'] + ['sibling_' + k for k, _ in GAPS] + ['sibling_unknown'] + ['uncle', 'grand', 'lineage_other'] + \
            ['sameuni_' + k for k, _ in GAPS] + ['sameuni_unknown'] + ['otheruni_' + k for k, _ in GAPS]
    X_rows = []; y_rows = []; cl = []; sib_rows = []; cells = collections.defaultdict(list)
    for i, e in enumerate(edges):
        a_ = e['professor_id']; u = e['university']; yi = int(e['year']) if str(e['year']).isdigit() else None
        g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = (students_of.get(g, set()) - {a_}) if g else set(); r = roots.get(a_)
        for j, pj in enumerate(pids):
            yj = phd_year.get(pj); gap = (yi - yj) if (yi is not None and yj is not None) else None
            same = bool(u) and (u in prof_unis[j])
            d = dict.fromkeys(names, 0.0); key = None
            if pj == a_:
                key = 'advisor_unknown' if gap is None else next('advisor_' + k for k, f in AGAPS if f(gap))
            elif pj in sibs:
                key = 'sibling_unknown' if gap is None else next('sibling_' + k for k, f in GAPS if f(gap))
                sib_rows.append({'edge_id': e['edge_id'], 'advisor': pinfo[a_]['name'], 'sibling_prof': pinfo[pj]['name'], 'thesis_year': yi, 'sibling_phd_year': yj, 'gap': gap, 'same_uni': int(same), 'sim_resid': round(float(S[i, j]), 5)})
            elif pj in uncles: key = 'uncle'
            elif g and pj == g: key = 'grand'
            elif root_size.get(r, 0) >= 2 and roots.get(pj) == r: key = 'lineage_other'
            elif same: key = 'sameuni_unknown' if gap is None else next('sameuni_' + k for k, f in GAPS if f(gap))
            else: key = None if gap is None else next('otheruni_' + k for k, f in GAPS if f(gap))  # 기준: 다른 대학·계보 밖·gap 미상
            if key: d[key] = 1.0
            cells[key].append(S[i, j])
            X_rows.append([d[k] for k in names]); y_rows.append(S[i, j]); cl.append(i)
    X = np.array(X_rows); y = np.array(y_rows); cl = np.array(cl)
    keep = X.sum(axis=0) > 0; names_k = [n for n, k in zip(names, keep) if k]; X = X[:, keep]
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1])); order = np.argsort(cl, kind='stable'); Xs, rs, cs = X[order], resid[order], cl[order]
    b = np.flatnonzero(np.diff(cs)) + 1; starts = np.concatenate([[0], b]); ends = np.concatenate([b, [len(cs)]])
    for s0, e0 in zip(starts, ends):
        sg = Xs[s0:e0].T @ rs[s0:e0]; meat += np.outer(sg, sg)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    reg = {n: {'beta': float(bb), 'se': float(s), 't': float(bb / s), 'n': int(len(cells[n]))} for n, bb, s in zip(names_k, beta, se)}
    reg['_note'] = f'기준 = 다른 대학·계보 밖·학위연도 미상 교수. 교수 학위 연도 확보 {cov}/{len(pids)}명. 논문 클러스터 SE.'
    json.dump(reg, open(out / f'cohort_regression{a.suffix}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(out / f'sibling_gap{a.suffix}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sib_rows[0].keys())); w.writeheader(); w.writerows(sib_rows)
    # 형제 쌍만: sim ~ |gap| 기울기
    sg = [(abs(r['gap']), r['sim_resid']) for r in sib_rows if r['gap'] is not None]
    slope = None
    if len(sg) > 20:
        xg = np.array([x for x, _ in sg], float); yg = np.array([y_ for _, y_ in sg]); A = np.column_stack([np.ones_like(xg), xg])
        coef = np.linalg.lstsq(A, yg, rcond=None)[0]; slope = {'intercept': float(coef[0]), 'slope_per_year': float(coef[1]), 'n': len(sg), 'corr': float(np.corrcoef(xg, yg)[0, 1])}
    lab = {'advisor': '지도교수(학위 후 경과 연수)', 'sibling': '형제 교수(논문 연도 − 형제 학위 연도)', 'sameuni': '같은 대학·계보 밖(연도 차)', 'otheruni': '다른 대학·계보 밖(연도 차)'}
    L = ['# 코호트 효과 분리 — 형제 교수 효과는 시기인가 계보인가', '', f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 논문 {len(edges)}편 × 교수 {len(pids)}명, 잔차 코사인. 교수 박사학위 연도 확보 {cov}/{len(pids)}명(학맥 간선·research-public).', '',
         '## 회귀 (기준: 다른 대학·계보 밖·연도 미상)', '| 항 | 계수 | SE | t | 쌍 수 |', '|---|---|---|---|---|']
    for n in names_k:
        grp = n.split('_')[0]; L.append(f"| {lab.get(grp, grp)} {n.split('_', 1)[1] if '_' in n else ''} | {reg[n]['beta']:+.4f} | {reg[n]['se']:.4f} | {reg[n]['t']:.1f} | {reg[n]['n']} |")
    if slope: L += ['', f"## 형제 쌍 안에서 연도 차와 유사도: 쌍 {slope['n']}개, 연도 차 1년당 {slope['slope_per_year']:+.5f} (r = {slope['corr']:+.3f}), 절편 {slope['intercept']:+.4f}"]
    L += ['', '## 읽는 법', '- 형제 교수 효과가 |연도 차| ≤5 에서만 크고 >15 에서 사라지면 "같은 시기 함께 배운 코호트" 효과이고, 멀어도 유지되면 지도교수를 통한 주제 전승이다.',
          '- 같은 대학·계보 밖 교수의 ≤5 항은 순수 코호트(다른 지도교수 아래 같은 시기 같은 대학원)다. 이것이 0 이면 코호트 자체는 닮음을 만들지 않는다.',
          '- 지도교수 항을 경과 연수로 나눈 것은 경력 단계별 전승 강도다. 연도 차 미상 항은 교수 학위 연도를 모르는 경우(도서관 노드 등).']
    (out / f'코호트분리_보고서{a.suffix}.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({n: (round(reg[n]['beta'], 4), round(reg[n]['t'], 1), reg[n]['n']) for n in names_k}, ensure_ascii=False)); print(slope)

if __name__ == '__main__':
    main()
