"""학풍 분석 통계 재검토 (2026-10-30, 연구자 지시 「총체적으로 재검토, 통계적 유의성 검정 포함」)

입력: prelim_similarity.py 출력 폴더(vectors.npz — stu·prof·pids·docs_vec·docs_pid·docs_year, vectors_meta.json), 저장소 genealogy-library.json, prof_phd_years.csv(선택)
종속변수: 제자 논문 × 교수 묶음 코사인에서 논문 평균·교수 평균을 뺀 잔차(이원 고정효과). 보고 단위는 코사인.
관계 더미(기준 = 다른 대학·계보 밖): 지도교수 / 조부 / 형제(같은 지도교수의 제자 교수) / 삼촌(조부의 다른 제자) / 같은 계보 기타 / 계보 밖 같은 대학 / 같은 대학×계보(지도교수 제외)
검정:
  1. OLS 계수 + 표준오차 세 가지 — 논문 클러스터 / 교수 클러스터 / **양방향 클러스터**(Cameron–Gelbach–Miller: V_논문 + V_교수 − V_HC0). 보고용 t·p 는 양방향.
  2. Holm 다중비교 보정(관계 항 7개).
  3. 클러스터 부트스트랩 — 교수(열) 재표집 / 논문(행) 재표집 각 B 회, 매 회 잔차화부터 다시 → 95% 백분위 구간.
  4. 순열 검정 — 같은 대학 교수들 사이에서 교수 정체(열)를 섞어(대학 구조는 보존, 계보 표지는 파괴) 관계 계수의 귀무분포.
  5. 효과 크기 — 계수 / 잔차 표준편차(표준화), 원 코사인 평균 차.
  6. 시계열 — 지도교수 묶음을 학위 연도 기준 창으로 나눠(학위 전 10년 / 학위 후 10년 / 11년 이후 위약) 창별 지도교수·형제 효과(창별 행렬을 따로 잔차화, 논문 부트스트랩 구간);
     학위 연대별(≤2004 / 2005–2014 / 2015–) 지도교수·형제·계보 밖 같은 대학 효과(양방향 클러스터 SE).
출력(--out): 통계검증_보고서.md, stat_review.json
"""
import argparse, json, csv, collections, time
from pathlib import Path
import numpy as np

REL = ['advisor', 'grand', 'sibling', 'uncle', 'lineage_other', 'same_uni_outside_lineage', 'same_uni_x_lineage']
LAB = {'advisor': '지도교수', 'grand': '조부 교수', 'sibling': '형제 교수', 'uncle': '삼촌 교수', 'lineage_other': '같은 계보 기타', 'same_uni_outside_lineage': '계보 밖 같은 대학', 'same_uni_x_lineage': '같은 대학×계보'}


def resid2(S, mask=None):
    """이원 고정효과 잔차. mask(결측)가 있으면 nan 무시 평균을 두 번 번갈아 뺀다."""
    if mask is None:
        return S - S.mean(axis=1, keepdims=True) - S.mean(axis=0, keepdims=True) + S.mean()
    R = np.where(mask, S, np.nan)
    for _ in range(3):
        R = R - np.nanmean(R, axis=1, keepdims=True); R = R - np.nanmean(R, axis=0, keepdims=True)
    return R


def ols(X, y):
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; return beta, XtX_inv, y - X @ beta


def cluster_V(X, resid, cl, XtX_inv):
    order = np.argsort(cl, kind='stable'); Xs, rs, cs = X[order], resid[order], cl[order]
    b = np.flatnonzero(np.diff(cs)) + 1; starts = np.concatenate([[0], b]); ends = np.concatenate([b, [len(cs)]])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for s0, e0 in zip(starts, ends):
        g = Xs[s0:e0].T @ rs[s0:e0]; meat += np.outer(g, g)
    return XtX_inv @ meat @ XtX_inv


def hc0_V(X, resid, XtX_inv):
    meat = (X * resid[:, None]).T @ (X * resid[:, None]); return XtX_inv @ meat @ XtX_inv


def pnorm2(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--phd-years'); ap.add_argument('--boot', type=int, default=300); ap.add_argument('--perm', type=int, default=300); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', default='')
    ap.add_argument('--edge-status', default='', help='교수→교수 간선을 student_identity.status(확정/추론/미확인) 쉼표 목록으로 제한 — 계보 판정 민감도 분석용. 비우면 전체 간선')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    P0 = Path(a.prelim); z = np.load(P0 / 'vectors.npz', allow_pickle=True); meta = json.load(open(P0 / 'vectors_meta.json', encoding='utf-8'))
    STU, PROF, pids = z['stu'].astype(np.float64), z['prof'].astype(np.float64), list(z['pids']); edges = meta['edges']; pinfo = meta['professors']
    n_e, n_p = len(edges), len(pids); pidx = {p: i for i, p in enumerate(pids)}
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    advisor_of = {}; allow_status = {x.strip() for x in a.edge_status.split(',') if x.strip()}; pp_edges = collections.Counter()
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']:
            st = (e.get('student_identity') or {}).get('status'); pp_edges['all'] += 1
            if allow_status and st not in allow_status: continue
            pp_edges['used'] += 1; advisor_of.setdefault(e['target'], e['source'])
    students_of = collections.defaultdict(set)
    for s, t in advisor_of.items(): students_of[t].add(s)
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
        return cur
    roots = {p: root(p) for p in set(pids) | set(advisor_of) | set(advisor_of.values())}; root_size = collections.Counter(roots[p] for p in pids)
    prof_unis = [set(pinfo[p]['universities']) for p in pids]
    # ---- 관계 유형 행렬 (0 기준, 1..7 = REL 순)
    R = np.zeros((n_e, n_p), dtype=np.int8); years = np.array([int(e['year']) if str(e['year']).isdigit() else -1 for e in edges])
    adv_col = np.zeros(n_e, dtype=int)
    for i, e in enumerate(edges):
        a_ = e['professor_id']; u = e['university']; g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = (students_of.get(g, set()) - {a_}) if g else set(); r = roots.get(a_)
        adv_col[i] = pidx[a_]
        for j, pj in enumerate(pids):
            same = bool(u) and (u in prof_unis[j])
            if pj == a_: k = 1
            elif g and pj == g: k = 2
            elif pj in sibs: k = 3
            elif pj in uncles: k = 4
            elif root_size.get(r, 0) >= 2 and roots.get(pj) == r: k = 5
            elif same: k = 6
            else: k = 0
            if k in (2, 3, 4, 5) and same: k = 7 + k   # 같은 대학×계보: 아래서 7번 더미와 함께 켠다
            R[i, j] = k
    def design(Rm):
        X = np.zeros(Rm.shape + (len(REL),)); base = Rm % 7 if False else None
        for k in range(1, 6): X[..., k - 1] = (Rm == k) | (Rm == 7 + k)
        X[..., 5] = (Rm == 6); X[..., 6] = (Rm >= 9)
        return X
    S_raw = STU @ PROF.T; S = resid2(S_raw)
    X = design(R).reshape(-1, len(REL)); y = S.reshape(-1); cl_i = np.repeat(np.arange(n_e), n_p); cl_j = np.tile(np.arange(n_p), n_e)
    keep = X.sum(axis=0) > 0; names = [n for n, k in zip(REL, keep) if k]; X = X[:, keep]
    beta, XtX_inv, res = ols(X, y)
    V_i = cluster_V(X, res, cl_i, XtX_inv); V_j = cluster_V(X, res, cl_j, XtX_inv); V_0 = hc0_V(X, res, XtX_inv); V_2 = V_i + V_j - V_0
    se_i, se_j, se_2 = np.sqrt(np.diag(V_i)), np.sqrt(np.diag(V_j)), np.sqrt(np.maximum(np.diag(V_2), 1e-18))
    t2 = beta / se_2; p2 = np.array([pnorm2(t) for t in t2]); p_holm = holm(p2)
    sd_res = float(y.std())
    raw_mean = {n: float(S_raw.reshape(-1)[X[:, k] == 1].mean()) for k, n in enumerate(names)}; raw_base = float(S_raw.reshape(-1)[X.sum(axis=1) == 0].mean())
    main_tab = {n: {'beta': float(beta[k]), 'se_thesis': float(se_i[k]), 'se_prof': float(se_j[k]), 'se_twoway': float(se_2[k]), 't_twoway': float(t2[k]), 'p_twoway': float(p2[k]), 'p_holm': float(p_holm[k]),
                    'std_effect': float(beta[k] / sd_res), 'n_pairs': int(X[:, k].sum()), 'raw_cos_mean': raw_mean[n], 'raw_cos_diff_vs_base': raw_mean[n] - raw_base} for k, n in enumerate(names)}
    # ---- 부트스트랩 (교수 열 / 논문 행)
    def fit_sub(rows, cols):
        Ss = resid2(S_raw[np.ix_(rows, cols)]); Xs = design(R[np.ix_(rows, cols)]).reshape(-1, len(REL))[:, keep]; ys = Ss.reshape(-1)
        if np.any(Xs.sum(axis=0) == 0): return None
        return np.linalg.lstsq(Xs, ys, rcond=None)[0]
    boots = {'prof': [], 'thesis': []}
    for _ in range(a.boot):
        b = fit_sub(np.arange(n_e), rng.integers(0, n_p, n_p))
        if b is not None: boots['prof'].append(b)
        b = fit_sub(rng.integers(0, n_e, n_e), np.arange(n_p))
        if b is not None: boots['thesis'].append(b)
    for kind in boots:
        B = np.array(boots[kind])
        for k, n in enumerate(names):
            main_tab[n][f'ci95_boot_{kind}'] = [float(np.percentile(B[:, k], 2.5)), float(np.percentile(B[:, k], 97.5))]; main_tab[n][f'se_boot_{kind}'] = float(B[:, k].std())
    # ---- 순열: 같은 대학 집단 안에서 교수 열 섞기
    strata = collections.defaultdict(list)
    for j in range(n_p): strata[tuple(sorted(prof_unis[j]))].append(j)
    perm_b = []
    for _ in range(a.perm):
        cols = np.arange(n_p)
        for g in strata.values():
            if len(g) > 1: cols[g] = rng.permutation(g)
        # 열을 섞되 S 는 그대로, 관계 표지만 섞인 교수에게서 가져온다 → 계보 표지 파괴, 대학 구조 보존
        Xp = design(R[:, cols]).reshape(-1, len(REL))[:, keep]
        if np.any(Xp.sum(axis=0) == 0): continue
        perm_b.append(np.linalg.lstsq(Xp, y, rcond=None)[0])
    PB = np.array(perm_b)
    for k, n in enumerate(names):
        main_tab[n]['perm_p'] = float((np.abs(PB[:, k]) >= abs(beta[k])).mean()) if len(PB) else None; main_tab[n]['perm_null_sd'] = float(PB[:, k].std()) if len(PB) else None
    # ---- 연대별 (지도교수·형제·계보 밖 같은 대학 × 연대)
    bins = [('≤2004', lambda yy: 0 < yy <= 2004), ('2005–2014', lambda yy: 2005 <= yy <= 2014), ('2015–', lambda yy: yy >= 2015)]
    decade = {}
    for lab, f in bins:
        rows = np.flatnonzero([f(yy) for yy in years])
        if len(rows) < 30: continue
        Ss = resid2(S_raw[rows]); Xs = design(R[rows]).reshape(-1, len(REL))[:, keep]; ys = Ss.reshape(-1)
        kk = Xs.sum(axis=0) > 0; Xs2 = Xs[:, kk]; nm2 = [n for n, q in zip(names, kk) if q]
        b_, XtXi_, r_ = ols(Xs2, ys); ci = np.repeat(np.arange(len(rows)), n_p); cj = np.tile(np.arange(n_p), len(rows))
        V = cluster_V(Xs2, r_, ci, XtXi_) + cluster_V(Xs2, r_, cj, XtXi_) - hc0_V(Xs2, r_, XtXi_); se_ = np.sqrt(np.maximum(np.diag(V), 1e-18))
        decade[lab] = {'n_theses': int(len(rows)), **{n: {'beta': float(b_[k]), 'se_twoway': float(se_[k]), 't': float(b_[k] / se_[k])} for k, n in enumerate(nm2) if n in ('advisor', 'sibling', 'uncle', 'grand', 'same_uni_outside_lineage')}}
    # ---- 시계열: 학위 연도 기준 교수 묶음 창
    windows = {}
    if 'docs_vec' in z.files:
        DV = z['docs_vec'].astype(np.float32); DP = z['docs_pid']; DY = z['docs_year']
        yrs = np.arange(1970, 2031); Y0 = yrs[0]
        # 교수×연도 누적합 (문서 벡터 합, 개수)
        cum = np.zeros((n_p, len(yrs) + 1, DV.shape[1]), dtype=np.float32); cnt = np.zeros((n_p, len(yrs) + 1), dtype=np.int32)
        for j in range(n_p):
            sel = (DP == j) & (DY >= Y0) & (DY <= yrs[-1])
            if sel.any():
                yy = DY[sel] - Y0; np.add.at(cum[j], yy + 1, DV[sel]); np.add.at(cnt[j], yy + 1, 1)
            cum[j] = np.cumsum(cum[j], axis=0); cnt[j] = np.cumsum(cnt[j])
        def win_mat(lo, hi):
            """각 간선 i 의 학위 연도 y 에 대해 교수 j 의 [y+lo, y+hi] 창 중심 벡터와의 코사인. 문서 3편 미만은 nan."""
            M = np.full((n_e, n_p), np.nan)
            for i in range(n_e):
                yv = years[i]
                if yv < 0: continue
                l_ = np.clip(yv + lo - Y0, 0, len(yrs)); h_ = np.clip(yv + hi - Y0 + 1, 0, len(yrs))
                if h_ <= l_: continue
                sums = cum[:, h_] - cum[:, l_]; ns = cnt[:, h_] - cnt[:, l_]
                ok = ns >= 3
                if not ok.any(): continue
                c = sums[ok] / np.linalg.norm(sums[ok], axis=1, keepdims=True).clip(1e-9); M[i, ok] = c @ STU[i]
            return M
        for lab, lo, hi in [('학위 전 10년 (−10…−1)', -10, -1), ('학위 전후 (−3…+3)', -3, 3), ('학위 후 10년 (+1…+10)', 1, 10), ('위약: 학위 11년 뒤 이후 (+11…+30)', 11, 30), ('위약: 학위 11년 전 이전 (−30…−11)', -30, -11)]:
            M = win_mat(lo, hi); mask = ~np.isnan(M); Rw = resid2(M, mask)
            rowmask = mask[np.arange(n_e), adv_col]
            vals = {}
            for k, n in [(1, 'advisor'), (3, 'sibling'), (6, 'same_uni_outside_lineage'), (0, 'base')]:
                sel = ((R == k) | (R == 7 + k if k in (2, 3, 4, 5) else False)) & mask
                vals[n] = {'mean_resid': float(np.nanmean(Rw[sel])) if sel.any() else None, 'n_pairs': int(sel.sum())}
            # 논문 부트스트랩 CI (지도교수 항)
            ii = np.flatnonzero(rowmask); adv_vals = Rw[ii, adv_col[ii]]; bs = [float(np.mean(rng.choice(adv_vals, len(adv_vals)))) for _ in range(a.boot)] if len(ii) >= 20 else []
            vals['advisor']['ci95_thesis_boot'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None; vals['advisor']['n_theses'] = int(len(ii))
            windows[lab] = vals
    res = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'label': a.label, 'prelim': str(P0), 'n_theses': n_e, 'n_professors': n_p, 'sd_resid': sd_res, 'boot': a.boot, 'perm': a.perm,
           'edge_status': sorted(allow_status) or 'all', 'prof_prof_edges': {'all': pp_edges['all'], 'used': pp_edges['used'], 'professors_with_advisor': len(advisor_of)},
           'main': main_tab, 'decade': decade, 'windows': windows}
    json.dump(res, open(out / 'stat_review.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # ---- 보고서
    L = [f'# 통계 재검토 — {a.label or P0.name}', '', f'생성 {res["generated"]}. 논문 {n_e}편 × 교수 {n_p}명, 종속변수 = 잔차 코사인(이원 고정효과), 잔차 SD {sd_res:.4f}. 부트스트랩 {a.boot}회, 순열 {a.perm}회. 계보 구성에 쓴 교수→교수 간선 {pp_edges["used"]}/{pp_edges["all"]}건' + (f'(판정 {",".join(sorted(allow_status))} 만)' if allow_status else '(전체)') + '.', '',
         '## 1. 관계별 계수와 표준오차 (기준 = 다른 대학·계보 밖)', '',
         '| 항 | 계수 | SE 논문 | SE 교수 | **SE 양방향** | t | p | p(Holm) | 순열 p | 교수 부트스트랩 95% | 논문 부트스트랩 95% | 표준화(β/SD) | 원 코사인 차 | 쌍 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n in names:
        m = main_tab[n]; cp = m['ci95_boot_prof']; ct = m['ci95_boot_thesis']
        L.append(f"| {LAB[n]} | {m['beta']:+.4f} | {m['se_thesis']:.4f} | {m['se_prof']:.4f} | **{m['se_twoway']:.4f}** | {m['t_twoway']:.1f} | {m['p_twoway']:.2g} | {m['p_holm']:.2g} | {m['perm_p'] if m['perm_p'] is None else format(m['perm_p'], '.3f')} | [{cp[0]:+.4f}, {cp[1]:+.4f}] | [{ct[0]:+.4f}, {ct[1]:+.4f}] | {m['std_effect']:+.3f} | {m['raw_cos_diff_vs_base']:+.4f} | {m['n_pairs']} |")
    L += ['', '읽는 법: 양방향 클러스터 SE 는 같은 논문·같은 교수가 여러 쌍에 반복 들어가는 상관을 둘 다 반영한다(논문 쪽만 잡은 앞 보고서의 SE 보다 크다). Holm 은 관계 항 7개의 다중비교 보정. 순열 p 는 같은 대학 교수끼리 정체를 섞어(계보 표지 파괴·대학 구조 보존) 얻은 양측 p — 0.000 은 순열 횟수 안에서 한 번도 넘지 못했다는 뜻. 표준화 효과는 잔차 SD 단위.', '',
          '## 2. 학위 연대별 (양방향 클러스터 SE)', '', '| 연대 | 논문 | 지도교수 | 형제 | 삼촌 | 조부 | 계보 밖 같은 대학 |', '|---|---|---|---|---|---|---|']
    for lab, d in decade.items():
        def c(n): return f"{d[n]['beta']:+.4f} ({d[n]['t']:.1f})" if n in d else '—'
        L.append(f"| {lab} | {d['n_theses']} | {c('advisor')} | {c('sibling')} | {c('uncle')} | {c('grand')} | {c('same_uni_outside_lineage')} |")
    L += ['', '## 3. 시계열 — 지도교수 묶음을 학위 연도 기준 창으로 나눔', '', '창마다 그 창의 문서가 3편 이상인 교수만 쓰고, 창별 행렬을 따로 잔차화한다. 수치 = 그 관계 쌍의 평균 잔차(기준 쌍 평균은 0 근처).', '',
          '| 창 | 논문 | 지도교수 | 95% (논문 부트스트랩) | 형제 | 계보 밖 같은 대학 | 기준(다른 대학·계보 밖) |', '|---|---|---|---|---|---|---|']
    for lab, v in windows.items():
        ad = v['advisor']; ci = ad.get('ci95_thesis_boot')
        def f(n): x = v[n]['mean_resid']; return ('—' if x is None else f"{x:+.4f}") + f" ({v[n]['n_pairs']})"
        L.append(f"| {lab} | {ad.get('n_theses', '')} | {f('advisor')} | {('[%+.4f, %+.4f]' % tuple(ci)) if ci else '—'} | {f('sibling')} | {f('same_uni_outside_lineage')} | {f('base')} |")
    L += ['', '읽는 법: 학위 **전** 10년 창의 지도교수 효과가 학위 **후** 창과 같거나 크면 "학생이 지도교수의 기존 연구를 이어받는" 전승이고, 후 창만 크면 역방향(지도교수가 제자 주제로 옮김) 또는 공동 진화다. 11년 이상 떨어진 위약 창에서도 효과가 남으면 "함께 변하는 주제"가 아니라 교수의 안정된 전공이 제자에게 전해진 것으로 읽는다.']
    (out / '통계검증_보고서.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({n: (round(main_tab[n]['beta'], 4), round(main_tab[n]['t_twoway'], 1), round(main_tab[n]['p_holm'], 4), main_tab[n]['perm_p']) for n in names}, ensure_ascii=False))
    print(json.dumps({k: {n: round(v[n]['mean_resid'], 4) if v[n]['mean_resid'] is not None else None for n in ('advisor', 'sibling', 'same_uni_outside_lineage')} for k, v in windows.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
