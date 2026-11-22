"""대학 학풍 검증 (2026-10-29) — prelim_similarity.py 가 저장한 임베딩(vectors.npz, vectors_meta.json)으로 계산한다.

1) within/between (제자 논문 기준): 제자 논문 i ↔ 지도교수를 뺀 같은 대학 교수들의 평균 코사인 − 다른 대학 교수들의 평균 코사인.
   귀무모형: 교수의 대학 표지를 섞음(1,000회).
2) within/between (교수 기준): 교수 프로필끼리 같은 대학 쌍 vs 다른 대학 쌍 평균 코사인, 같은 순열.
3) 논문 고정효과 회귀: 모든 (논문 i, 교수 j) 쌍에서 sim_ij − mean_i = β1·같은 지도교수 + β2·같은 대학(지도교수 아님) + β3·교수 문서 수(log) ; 논문 단위 클러스터 표준오차.
   교수 j 가 여러 대학 소속이면 그중 하나가 논문 대학과 같을 때 '같은 대학'.
4) 대학 식별: 제자 논문 벡터로 대학을 맞히기 — 교수 프로필 최근접(지도교수 제외, 교수 k=1 과 대학 평균 k=5 투표), 우연(=대학 비율) 과 비교. 대학별 재현율.
5) 세대 전승: 지도교수의 지도교수(조부 교수)가 교수 묶음에 있으면 sim(논문, 조부) vs 같은 대학 다른 교수 평균.
6) 민감도: 미확인 간선 제외.
출력(--out): within_between_students.csv, within_between_professors.json, regression.json, university_identification.csv, lineage_decay.csv, 학풍검증_보고서.md
"""
import argparse, json, csv, collections, math, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', required=True, help='학풍_예비분석 폴더'); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--perm', type=int, default=1000); ap.add_argument('--min-edges', type=int, default=10); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--raw', action='store_true', help='교수·논문 고정효과를 빼지 않은 원 코사인으로 계산(참고용)')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    P0 = Path(a.prelim); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(P0 / 'vectors.npz', allow_pickle=True); meta = json.load(open(P0 / 'vectors_meta.json', encoding='utf-8'))
    STU, PROF, pids = z['stu'], z['prof'], list(z['pids']); ndocs = z['prof_ndocs'].astype(float)
    edges = meta['edges']; pinfo = meta['professors']
    S_raw = STU @ PROF.T  # (논문 × 교수)
    # 교수 묶음이 클수록 센트로이드가 '일반적'이 되어 모든 논문과 코사인이 높다(문서 수와 상관). 교수 고정효과·논문 고정효과를 뺀 잔차로 분석하고 원값은 참고로 둔다.
    S_res = S_raw - S_raw.mean(axis=1, keepdims=True) - S_raw.mean(axis=0, keepdims=True) + S_raw.mean()
    S = S_res if not a.raw else S_raw
    pidx = {p: i for i, p in enumerate(pids)}
    adv = np.array([pidx[e['professor_id']] for e in edges])
    uni_e = [e['university'] for e in edges]
    prof_unis = [set(pinfo[p]['universities']) for p in pids]
    unis_all = sorted({u for u in uni_e if u}); U = len(unis_all)
    same_uni = np.zeros(S.shape, dtype=bool)  # (i, j): 교수 j 가 논문 i 의 대학 소속
    for i, u in enumerate(uni_e):
        if u: same_uni[i] = np.array([u in pu for pu in prof_unis])
    is_adv = np.zeros(S.shape, dtype=bool); is_adv[np.arange(len(edges)), adv] = True
    corr_nd = float(np.corrcoef(S_raw.mean(axis=0), np.log(ndocs))[0, 1])
    L = ['# 학풍 검증 보고 — 지도교수 효과를 뺀 대학 효과', '', f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 논문 {len(edges)}편 × 교수 {len(pids)}명. 유사도 = bge-m3 제목 임베딩 코사인' + (' **원값**' if a.raw else ' **잔차**(논문 고정효과·교수 고정효과 제거)') + '.',
         f'- 원 코사인은 교수 묶음 크기와 상관이 있다(교수별 평균 코사인 ↔ log 문서 수 r = {corr_nd:.2f}): 문서가 많은 교수의 센트로이드가 모든 논문과 가까워진다. 그래서 이 보고의 수치는 교수·논문 평균을 뺀 잔차다(`--raw` 로 원값 보고 생성).', '']

    # ---- 1) 제자 논문 기준 within/between
    def wb_students(S, same_uni, is_adv):
        w = np.where(same_uni & ~is_adv, S, np.nan); b = np.where(~same_uni & ~is_adv, S, np.nan)
        return np.nanmean(w, axis=1), np.nanmean(b, axis=1)
    win, btw = wb_students(S, same_uni, is_adv); valid = ~np.isnan(win) & ~np.isnan(btw)
    d = (win - btw)[valid]; obs = float(d.mean())
    # 귀무: 교수 대학 표지 섞기 (교수 단위)
    null = []
    for _ in range(a.perm):
        perm_idx = rng.permutation(len(pids)); su = same_uni[:, perm_idx]; ia = is_adv  # 지도교수 자리는 그대로 제외
        w2, b2 = wb_students(S, su, ia); v2 = ~np.isnan(w2) & ~np.isnan(b2)
        null.append(float((w2 - b2)[v2].mean()))
    null = np.array(null); zs = (obs - null.mean()) / (null.std() + 1e-12)
    L += ['## 1. 제자 논문 ↔ (지도교수를 뺀) 같은 대학 교수 vs 다른 대학 교수',
          f'- 평균 차이(같은 대학 − 다른 대학) = **{obs:+.4f}** (논문 {int(valid.sum())}편). 교수의 대학 표지를 섞은 귀무분포 평균 {null.mean():+.4f} (sd {null.std():.4f}) → z = {zs:.2f}, p = {(null >= obs).mean():.3f}',
          f'- 같은 대학 교수와의 평균 코사인 {np.nanmean(win):.4f}, 다른 대학 교수와의 평균 {np.nanmean(btw):.4f}. 지도교수 본인과의 코사인은 {S[np.arange(len(edges)), adv].mean():.4f}.', '']
    by_u = collections.defaultdict(list)
    for i in np.where(valid)[0]: by_u[uni_e[i]].append(i)
    rows_u = []
    L += ['| 대학 | 논문 | 같은 대학 교수(지도교수 제외) | 다른 대학 교수 | 차이 | 순열 z | p |', '|---|---|---|---|---|---|---|']
    for u in sorted(by_u, key=lambda x: -len(by_u[x])):
        ii = np.array(by_u[u])
        if len(ii) < a.min_edges: continue
        o = float((win - btw)[ii].mean()); nu = []
        for _ in range(min(a.perm, 500)):
            perm_idx = rng.permutation(len(pids)); su = same_uni[:, perm_idx]
            w2, b2 = wb_students(S[ii], su[ii], is_adv[ii]); v2 = ~np.isnan(w2) & ~np.isnan(b2)
            nu.append(float((w2 - b2)[v2].mean()) if v2.any() else np.nan)
        nu = np.array([x for x in nu if not np.isnan(x)])
        zz = (o - nu.mean()) / (nu.std() + 1e-12) if len(nu) else float('nan'); pp = float((nu >= o).mean()) if len(nu) else float('nan')
        rows_u.append({'university': u, 'n_theses': len(ii), 'within_mean': round(float(win[ii].mean()), 4), 'between_mean': round(float(btw[ii].mean()), 4), 'diff': round(o, 4), 'perm_z': round(zz, 2), 'perm_p': round(pp, 3)})
        L.append(f'| {u} | {len(ii)} | {win[ii].mean():.4f} | {btw[ii].mean():.4f} | {o:+.4f} | {zz:.2f} | {pp:.3f} |')
    with open(out / 'within_between_students.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['university', 'n_theses', 'within_mean', 'between_mean', 'diff', 'perm_z', 'perm_p']); w.writeheader(); w.writerows(rows_u)

    # ---- 2) 교수 기준 within/between
    PP = PROF @ PROF.T; n = len(pids)
    same_pp = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (prof_unis[i] & prof_unis[j]): same_pp[i, j] = True
    diff_pp = ~np.eye(n, dtype=bool) & ~same_pp
    obs_pp = float(PP[same_pp].mean() - PP[diff_pp].mean()); null_pp = []
    for _ in range(a.perm):
        perm_idx = rng.permutation(n); sp = same_pp[np.ix_(perm_idx, perm_idx)]; dp = ~np.eye(n, dtype=bool) & ~sp
        null_pp.append(float(PP[sp].mean() - PP[dp].mean()))
    null_pp = np.array(null_pp); z_pp = (obs_pp - null_pp.mean()) / (null_pp.std() + 1e-12)
    res_pp = {'same_university_pairs': int(same_pp.sum() // 2), 'different_pairs': int(diff_pp.sum() // 2), 'mean_cos_same': float(PP[same_pp].mean()), 'mean_cos_diff': float(PP[diff_pp].mean()),
              'diff': obs_pp, 'perm_mean': float(null_pp.mean()), 'perm_sd': float(null_pp.std()), 'z': float(z_pp), 'p': float((null_pp >= obs_pp).mean())}
    json.dump(res_pp, open(out / 'within_between_professors.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    L += ['', '## 2. 교수 프로필끼리: 같은 대학 쌍 vs 다른 대학 쌍',
          f"- 같은 대학 교수 쌍 {res_pp['same_university_pairs']}개 평균 코사인 {res_pp['mean_cos_same']:.4f}, 다른 대학 쌍 {res_pp['different_pairs']}개 {res_pp['mean_cos_diff']:.4f} → 차이 **{obs_pp:+.4f}**, 대학 표지 순열 z = {z_pp:.2f}, p = {res_pp['p']:.3f}", '']

    # ---- 3) 논문 고정효과 회귀 (클러스터 SE)
    Sd = S - S.mean(axis=1, keepdims=True)
    X1 = is_adv.astype(float); X2 = (same_uni & ~is_adv).astype(float); X3 = np.broadcast_to(np.log(ndocs)[None, :], S.shape); X3 = X3 - X3.mean(axis=1, keepdims=True)
    y = Sd.ravel(); X = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel()]); cl = np.repeat(np.arange(len(edges)), len(pids))
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; resid = y - X @ beta
    meat = np.zeros((3, 3))
    for g in range(len(edges)):
        sl = slice(g * len(pids), (g + 1) * len(pids)); xg = X[sl]; ug = resid[sl]; s = xg.T @ ug; meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv; se = np.sqrt(np.diag(V))
    reg = {'n_pairs': int(len(y)), 'n_theses': len(edges), 'n_professors': len(pids),
           'beta_same_advisor': float(beta[0]), 'se': float(se[0]), 't': float(beta[0] / se[0]),
           'beta_same_university_not_advisor': float(beta[1]), 'se_uni': float(se[1]), 't_uni': float(beta[1] / se[1]),
           'beta_log_ndocs': float(beta[2]), 'se_ndocs': float(se[2]), 'note': '종속변수 = 코사인 − 논문별 평균(논문 고정효과). 표준오차는 논문 단위 클러스터.'}
    json.dump(reg, open(out / 'regression.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    L += ['## 3. 고정효과 회귀 (모든 논문×교수 쌍 %d개)%s' % (len(y), '' if a.raw else ' — 종속변수가 이미 교수 고정효과까지 뺀 잔차라 문서 수 항은 0 에 가깝다'), '| 항 | 계수 | 클러스터 SE | t |', '|---|---|---|---|',
          f"| 같은 지도교수 | {beta[0]:+.4f} | {se[0]:.4f} | {beta[0]/se[0]:.1f} |", f"| 같은 대학(지도교수 아님) | {beta[1]:+.4f} | {se[1]:.4f} | {beta[1]/se[1]:.1f} |", f"| 교수 문서 수(log, 논문 안 중심화) | {beta[2]:+.4f} | {se[2]:.4f} | {beta[2]/se[2]:.1f} |",
          '', f'- 지도교수 효과 대비 대학 효과 비율 = {beta[1]/beta[0]:.2f}' if beta[0] else '', '']

    # ---- 4) 대학 식별
    uni_idx = {u: k for k, u in enumerate(unis_all)}
    rows_id = []; correct1 = 0; correct5 = 0; nvalid = 0; recall = collections.Counter(); tot = collections.Counter(); chance = collections.Counter()
    for i, e in enumerate(edges):
        u = uni_e[i]
        if not u: continue
        sims = S[i].copy(); sims[adv[i]] = -9
        order = np.argsort(-sims)
        top1 = next((pu for pu in [prof_unis[order[0]]] if pu), set()); guess1 = sorted(top1)[0] if top1 else ''
        votes = collections.Counter()
        for j in order[:5]:
            for pu in prof_unis[j]: votes[pu] += float(sims[j])
        guess5 = votes.most_common(1)[0][0] if votes else ''
        nvalid += 1; tot[u] += 1
        if u in top1: correct1 += 1; recall[u] += 1
        if guess5 == u: correct5 += 1
        rows_id.append({'edge_id': e['edge_id'], 'university': u, 'guess_top1_prof_uni': '|'.join(sorted(top1)), 'guess_top5_vote': guess5, 'hit_top1': int(u in top1), 'hit_top5vote': int(guess5 == u)})
    share = {u: tot[u] / nvalid for u in tot}
    # 우연 = 논문마다 (지도교수를 뺀 교수 가운데 같은 대학 교수 비율)의 평균 — 최근접 교수를 무작위로 뽑았을 때 맞힐 확률
    chance_acc = float(np.mean([(same_uni[i].sum() - 1) / (len(pids) - 1) for i in range(len(edges)) if uni_e[i]]))
    with open(out / 'university_identification.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_id[0].keys())); w.writeheader(); w.writerows(rows_id)
    L += ['## 4. 제자 논문만으로 대학 맞히기 (지도교수 제외, 교수 프로필 최근접)',
          f'- 최근접 교수의 대학 = 논문 대학: **{correct1/nvalid:.3f}**, 상위 5교수 가중 투표: {correct5/nvalid:.3f}. 우연(지도교수를 뺀 교수 중 같은 대학 비율의 평균) {chance_acc:.3f}; 참고로 최대 대학만 찍으면 {max(share.values()):.3f}.',
          '| 대학 | 논문 | 재현율(최근접 교수 대학) |', '|---|---|---|']
    for u in sorted(tot, key=lambda x: -tot[x]):
        if tot[u] >= a.min_edges: L.append(f'| {u} | {tot[u]} | {recall[u]/tot[u]:.3f} |')

    # ---- 5) 세대 전승
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8'))
    advisor_of = {}
    for e in gl['edges']:
        if e.get('target') in pidx and e.get('source') in pidx: advisor_of[e['target']] = e['source']
    rows_g = []
    for i, e in enumerate(edges):
        gp = advisor_of.get(e['professor_id'])
        if not gp: continue
        g = pidx[gp]; mask = same_uni[i] & ~is_adv[i]; mask[g] = False
        same_other = float(np.nanmean(np.where(mask, S[i], np.nan))) if mask.any() else float('nan')
        rows_g.append({'edge_id': e['edge_id'], 'university': uni_e[i], 'advisor': pinfo[e['professor_id']]['name'], 'grand_advisor': pinfo[gp]['name'], 'sim_advisor': round(float(S[i, adv[i]]), 4),
                       'sim_grand_advisor': round(float(S[i, g]), 4), 'sim_same_uni_others': round(same_other, 4) if not math.isnan(same_other) else '', 'sim_all_others_mean': round(float(np.delete(S[i], [adv[i], g]).mean()), 4)})
    if rows_g:
        with open(out / 'lineage_decay.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_g[0].keys())); w.writeheader(); w.writerows(rows_g)
        ga = np.array([r['sim_grand_advisor'] for r in rows_g]); aa = np.array([r['sim_advisor'] for r in rows_g]); oo = np.array([r['sim_all_others_mean'] for r in rows_g])
        so = np.array([r['sim_same_uni_others'] for r in rows_g if r['sim_same_uni_others'] != ''], dtype=float)
        L += ['', '## 5. 세대 전승: 제자 논문 ↔ 조부 교수(지도교수의 지도교수)', f'- 조부 교수가 묶음에 있는 논문 {len(rows_g)}편. 지도교수 {aa.mean():.4f} > 조부 교수 **{ga.mean():.4f}** > 같은 대학 다른 교수 {so.mean() if len(so) else float("nan"):.4f} > 전체 다른 교수 {oo.mean():.4f}.',
              f'- 조부 교수 − 전체 다른 교수 = {ga.mean()-oo.mean():+.4f} (부호검정: 조부가 더 높은 논문 {(ga > oo).mean():.2%}).']
        gpairs = collections.Counter((r['grand_advisor'], r['advisor']) for r in rows_g)
        L += ['', '| 조부 교수 → 지도교수 | 논문 | 지도교수 | 조부 | 같은 대학 다른 교수 |', '|---|---|---|---|---|']
        for (g_, a_), c in gpairs.most_common(15):
            rs = [r for r in rows_g if r['grand_advisor'] == g_ and r['advisor'] == a_]; so2 = [r['sim_same_uni_others'] for r in rs if r['sim_same_uni_others'] != '']
            L.append(f"| {g_} → {a_} | {c} | {np.mean([r['sim_advisor'] for r in rs]):.4f} | {np.mean([r['sim_grand_advisor'] for r in rs]):.4f} | {np.mean(so2) if so2 else float('nan'):.4f} |")

    # ---- 6) 민감도: 미확인 제외
    keep = np.array([e['identity'] != '미확인' for e in edges])
    if keep.sum() < len(edges):
        w3, b3 = wb_students(S[keep], same_uni[keep], is_adv[keep]); v3 = ~np.isnan(w3) & ~np.isnan(b3)
        L += ['', '## 6. 민감도 — 동일인 미확인 간선 제외', f'- 남은 논문 {int(keep.sum())}편: 같은 대학 − 다른 대학 = {(w3-b3)[v3].mean():+.4f}; 지도교수 코사인 평균 {S[keep][np.arange(int(keep.sum())), adv[keep]].mean():.4f}']
    L += ['', '## 읽는 법', '- 1·3 의 "같은 대학" 효과가 양이고 유의하면 지도교수를 넘어서는 학교 수준의 닮음(학풍)이 있다. 0 에 가까우면 학풍은 지도교수 효과의 합으로 설명된다.',
          '- 2 는 교수들 자신이 같은 대학 안에서 닮았는지(교수 채용·학맥의 결과)를 본다. 1 과 2 가 함께 양이면 학풍이 제자 세대까지 전달된다는 뜻.',
          '- 4 의 대학 재현율은 그 대학 교수 수에 비례해 오르므로 대학 간 비교보다 전체 정확도 vs 우연을 본다.',
          '- 모든 수치는 수집 자료 안의 연결(교수 묶음 = 국중 저작 + 소속 일치 KCI 논문)만 센 값이다.']
    (out / ('학풍검증_보고서_원값.md' if a.raw else '학풍검증_보고서.md')).write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({'wb_students_diff': obs, 'z': float(zs), 'wb_prof': res_pp['diff'], 'z_prof': res_pp['z'], 'beta_adv': reg['beta_same_advisor'], 'beta_uni': reg['beta_same_university_not_advisor'], 't_uni': reg['t_uni'], 'uni_id_top1': correct1 / nvalid, 'chance': chance_acc, 'grand_pairs': len(rows_g)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
