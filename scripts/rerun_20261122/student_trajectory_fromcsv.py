"""제자 쪽 시계열 — 학위 뒤 본인 KCI 논문으로 본 이탈 속도와 재직 대학 동료 쪽 이동 (2026-10-30, 연구자 「이어서 하자」)

질문 1  제자는 학위 뒤 몇 년에 걸쳐 지도교수 주제에서 멀어지는가? (지도교수 묶음과의 잔차 코사인, 본인 박사논문과의 코사인을 학위 후 경과 연수 구간별로)
질문 2  재직 대학(논문 발표 당시 소속)의 동료 교수 쪽으로 옮겨 가는가? 학위 대학에 남은 사람과 옮긴 사람이 다른가?
자료   학위자·교수 A 등급 cretId 집합(cret_best.csv)의 KCI 논문(제목·키워드·초록 600자, bge-m3) × 교수 묶음(학풍_A기준_KCI만_전체 vectors.npz 의 교수 중심 벡터·문서 벡터)
방법   논문 × 교수 코사인 행렬을 논문·교수 고정효과로 잔차화(본인이 교수 열이면 그 칸은 결측). 관계: 지도교수 / 계보(조부·형제·삼촌·기타) / 재직 대학 동료(계보 밖) / 학위 대학 동료(계보 밖, 재직 대학과 다를 때) / 그 외.
       구간 = 발표 연도 − 학위 연도: 학위 전(−5…−1), 0, 1–3, 4–6, 7–10, 11–15, 16+.
       회귀: 잔차 = Σ(관계 × 구간) 더미, 기준 = 그 외 교수. SE 는 제자(사람)·교수 양방향 클러스터. 구간별 평균은 제자 재표집 부트스트랩 95%.
       지도교수 '학위 전 10년' 창 중심(물려받은 주제)과의 잔차도 따로 — 교수 본인이 뒤에 주제를 바꿔도 "물려받은 주제"에서의 이탈을 본다.
출력(--out): student_articles.csv(논문별, 초록 없음), trajectory.json, 제자궤적_보고서.md
"""
import argparse, csv, json, re, sys, time, collections, unicodedata
from pathlib import Path
import numpy as np

BINS = [('학위 전(−5…−1)', -5, -1), ('0', 0, 0), ('1–3', 1, 3), ('4–6', 4, 6), ('7–10', 7, 10), ('11–15', 11, 15), ('16+', 16, 60)]


def nm(s):
    s = unicodedata.normalize('NFKC', str(s or '')); return re.sub(r'[\s,.·\-]+', '', s)


def resid2(S, mask):
    R = np.where(mask, S, np.nan)
    for _ in range(3):
        R = R - np.nanmean(R, axis=1, keepdims=True); R = R - np.nanmean(R, axis=0, keepdims=True)
    return R


def ols_2way(X, y, cl1, cl2):
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y; res = y - X @ beta
    def clV(cl):
        order = np.argsort(cl, kind='stable'); Xs, rs, cs = X[order], res[order], cl[order]
        b = np.flatnonzero(np.diff(cs)) + 1; st = np.concatenate([[0], b]); en = np.concatenate([b, [len(cs)]])
        meat = np.zeros((X.shape[1], X.shape[1]))
        for s0, e0 in zip(st, en):
            g = Xs[s0:e0].T @ rs[s0:e0]; meat += np.outer(g, g)
        return XtX_inv @ meat @ XtX_inv
    V0 = XtX_inv @ ((X * res[:, None]).T @ (X * res[:, None])) @ XtX_inv
    V = clV(cl1) + clV(cl2) - V0; se = np.sqrt(np.maximum(np.diag(V), 1e-18)); return beta, se


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prelim', required=True, help='학풍_A기준_KCI만_전체 폴더(vectors.npz 에 docs_vec 필요)'); ap.add_argument('--repo', required=True); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=300); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--no-embed', action='store_true'); ap.add_argument('--docs-from', help='기존 student_articles.csv 에서 논문 목록을 읽는다(cret_best.csv·author_ids.csv 를 쓰지 않음; 임베딩 캐시 student_vecs.npz 필요)')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    repo, base, P0 = Path(a.repo), Path(a.base), Path(a.prelim)
    sys.path.insert(0, str(repo))
    try:
        from hanja_norm import uni_norm
    except Exception:   # venv 에 hanja 패키지가 없을 때: KCI 소속은 한글이라 간단 정규화로 충분
        def uni_norm(v):
            s = re.sub(r'^국립\s*', '', str(v or '').strip()); s = re.split(r'\s', s)[0] if s else ''
            s = re.sub(r'(대학원|대학교|대학)$', '대', s) if not s.endswith('대') else s
            return s.replace('여자대', '여대')
    z = np.load(P0 / 'vectors.npz', allow_pickle=True); meta = json.load(open(P0 / 'vectors_meta.json', encoding='utf-8'))
    STU, PROF, pids = z['stu'].astype(np.float32), z['prof'].astype(np.float32), list(z['pids']); edges = meta['edges']; pinfo = meta['professors']
    DV, DP, DY = z['docs_vec'].astype(np.float32), z['docs_pid'], z['docs_year']
    n_p = len(pids); pidx = {p: i for i, p in enumerate(pids)}; prof_unis = [set(pinfo[p]['universities']) for p in pids]
    # ---- 학맥 관계
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    advisor_of = {}
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']: advisor_of.setdefault(e['target'], e['source'])
    students_of = collections.defaultdict(set)
    for s, t in advisor_of.items(): students_of[t].add(s)
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
        return cur
    roots = {p: root(p) for p in set(pids) | set(advisor_of) | set(advisor_of.values())}; root_size = collections.Counter(roots[p] for p in pids)
    if a.docs_from:
        eidx = {e['edge_id']: i for i, e in enumerate(edges)}; docs = []
        for r in csv.DictReader(open(a.docs_from, encoding='utf-8-sig', newline='')):
            docs.append({'edge_i': eidx[r['edge_id']], 'kci_id': r['kci_id'], 'year': int(r['year']), 'dt': int(r['dt']), 'inst': r['institution'], 'text': '', 'label': r['title']})
        print('docs from csv', len(docs), 'students', len({d['edge_i'] for d in docs}), flush=True)
        n_students = len({d['edge_i'] for d in docs}); no_cret = len(edges) - 905   # 원 실행 기록(A 등급 cretId 있는 학위자 905명) — CSV 모드에서는 다시 세지 않는다
    else:
        # ---- 학위자 → cretId (A)
        cret_of = {}
        for r in csv.DictReader(open(base / '참고문헌_구축/저자ID/식별/cret_best.csv', encoding='utf-8-sig', newline='')):
            if r['grade'] == 'A': cret_of[r['person_id']] = [x for x in (r.get('cret_ids_A') or r['cret_id']).split('; ') if x]
        arts = {}
        for r in csv.DictReader(open(base / '참고문헌_API/source_articles.csv', encoding='utf-8-sig', newline='')):
            y = r['year'][:4]
            arts[r['kci_id']] = {'title': r['title'], 'kw': (r.get('keywords_english') or '').replace(';', ', '), 'abs': re.sub(r'\s+', ' ', r.get('abstract_ko', '')).strip()[:600], 'year': int(y) if y.isdigit() else None}
        cret_arts = collections.defaultdict(list)
        for r in csv.DictReader(open(base / '참고문헌_구축/저자ID/author_ids.csv', encoding='utf-8-sig', newline='')):
            if r['cret_id']: cret_arts[r['cret_id']].append((r['kci_id'], uni_norm(r['institution']) if r['institution'] else ''))
        # ---- 간선(제자 논문)별 본인 KCI 논문 모으기
        docs = []   # dict(edge_i, kci_id, year, dt, inst, text, label)
        n_students = 0; no_cret = 0
        for i, e in enumerate(edges):
            y0 = int(e['year']) if str(e['year']).isdigit() else None
            if y0 is None: continue
            crets = cret_of.get(e['thesis_id']) or cret_of.get(e['student']) or []
            if not crets: no_cret += 1; continue
            seen = set(); any_doc = False
            for c in crets:
                for aid, inst in cret_arts.get(c, []):
                    art = arts.get(aid)
                    if not art or art['year'] is None or aid in seen: continue
                    seen.add(aid); dt = art['year'] - y0
                    if dt < -5: continue
                    text = art['title'] + ((' — ' + art['kw']) if art['kw'] else '') + (('\n' + art['abs']) if art['abs'] else '')
                    docs.append({'edge_i': i, 'kci_id': aid, 'year': art['year'], 'dt': dt, 'inst': inst, 'text': text, 'label': art['title']}); any_doc = True
            n_students += any_doc
        print('edges', len(edges), 'students with cretId', len(edges) - no_cret, 'with articles', n_students, 'articles', len(docs), flush=True)
    # ---- 임베딩
    cache = out / 'student_vecs.npz'; AV = None
    if cache.exists():
        zc = np.load(cache, allow_pickle=True)
        if list(zc['kci_ids']) == [d['kci_id'] for d in docs]: AV = zc['vecs'].astype(np.float32); print('embeddings reused', AV.shape, flush=True)
    if AV is None:
        import torch  # noqa
        sys.path.insert(0, r'<HOME>\Downloads\drive-download-20260918T092441Z-1-002\작업도구\추천')
        from embed_local_bge import LocalBGE
        model = LocalBGE(max_len=512, threads=4); t0 = time.time(); vecs = []
        texts = [d['text'] for d in docs]
        for k in range(0, len(texts), 256): vecs.append(np.asarray(model.encode(texts[k:k + 256], batch=64), dtype=np.float32))
        AV = np.vstack(vecs); AV /= np.linalg.norm(AV, axis=1, keepdims=True) + 1e-9; print('embedded', AV.shape, f'{time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(cache, vecs=AV.astype(np.float16), kci_ids=np.array([d['kci_id'] for d in docs]))
    # ---- '학위 전 10년' 창: 모든 교수의 같은 창 중심(문서 3편 이상) — 물려받은 주제와의 잔차를 다른 교수의 같은 시기 주제와 비교(이원 잔차)
    yrs = np.arange(1970, 2031); Y0 = yrs[0]
    cum = np.zeros((n_p, len(yrs) + 1, DV.shape[1]), dtype=np.float32); cnt = np.zeros((n_p, len(yrs) + 1), dtype=np.int32)
    for j in range(n_p):
        sel = (DP == j) & (DY >= Y0) & (DY <= yrs[-1])
        if sel.any():
            yy = DY[sel] - Y0; np.add.at(cum[j], yy + 1, DV[sel]); np.add.at(cnt[j], yy + 1, 1)
        cum[j] = np.cumsum(cum[j], axis=0); cnt[j] = np.cumsum(cnt[j])
    win_cache = {}
    def pre_window(y0):
        if y0 not in win_cache:
            l_ = np.clip(y0 - 10 - Y0, 0, len(yrs)); h_ = np.clip(y0 - 1 - Y0 + 1, 0, len(yrs))
            sums = cum[:, h_] - cum[:, l_]; ns = cnt[:, h_] - cnt[:, l_]; ok = ns >= 3
            W = np.full((n_p, DV.shape[1]), np.nan, dtype=np.float32)
            if h_ > l_ and ok.any(): W[ok] = sums[ok] / np.linalg.norm(sums[ok], axis=1, keepdims=True).clip(1e-9)
            win_cache[y0] = (W, ok)
        return win_cache[y0]
    # ---- 논문 × 교수 행렬, 관계 코드
    n_a = len(docs); S = AV @ PROF.T
    adv_col = np.array([pidx[edges[d['edge_i']]['professor_id']] for d in docs])
    adv_col = np.array([pidx[edges[d['edge_i']]['professor_id']] for d in docs])
    stu_node = [edges[d['edge_i']]['student'] for d in docs]
    self_col = np.array([pidx.get(s, -1) for s in stu_node])
    mask = np.ones_like(S, dtype=bool)
    for r_, j in enumerate(self_col):
        if j >= 0: mask[r_, j] = False
    R = resid2(S, mask)
    # 관계 코드: 1 지도교수, 2 조부, 3 형제, 4 삼촌, 5 계보 기타, 6 재직 대학 동료(계보 밖), 7 학위 대학 동료(계보 밖, 재직≠학위), 0 그 외
    REL = {1: '지도교수', 2: '조부', 3: '형제', 4: '삼촌', 5: '계보 기타', 6: '재직 대학 동료', 7: '학위 대학 동료'}
    K = np.zeros((n_a, n_p), dtype=np.int8); moved = np.zeros(n_a, dtype=bool)
    for r_, d in enumerate(docs):
        e = edges[d['edge_i']]; a_ = e['professor_id']; g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = (students_of.get(g, set()) - {a_}) if g else set(); rt = roots.get(a_)
        deg_u = e['university']; cur_u = d['inst']; moved[r_] = bool(cur_u) and bool(deg_u) and (cur_u != deg_u)
        for j, pj in enumerate(pids):
            if pj == a_: k = 1
            elif g and pj == g: k = 2
            elif pj in sibs: k = 3
            elif pj in uncles: k = 4
            elif root_size.get(rt, 0) >= 2 and roots.get(pj) == rt: k = 5
            elif cur_u and cur_u in prof_unis[j]: k = 6
            elif deg_u and deg_u in prof_unis[j] and cur_u != deg_u: k = 7
            else: k = 0
            K[r_, j] = k
    dts = np.array([d['dt'] for d in docs]); bin_of = np.full(n_a, -1)
    for b, (_, lo, hi) in enumerate(BINS): bin_of[(dts >= lo) & (dts <= hi)] = b
    # ---- 논문별 지표
    adv_res = R[np.arange(n_a), adv_col]
    own_thesis = np.einsum('ij,ij->i', AV, STU[[d['edge_i'] for d in docs]])
    S_pre = np.full((n_a, n_p), np.nan, dtype=np.float32)
    for r_, d in enumerate(docs):
        y0 = int(edges[d['edge_i']]['year']); W, ok = pre_window(y0)
        if ok.any(): S_pre[r_, ok] = W[ok] @ AV[r_]
    mask_pre = ~np.isnan(S_pre) & mask
    R_pre = resid2(S_pre, mask_pre)
    adv_pre_res = np.where(mask_pre[np.arange(n_a), adv_col], R_pre[np.arange(n_a), adv_col], np.nan)
    def rel_mean(r_, k):
        sel = (K[r_] == k) & mask[r_]; return float(np.nanmean(R[r_, sel])) if sel.any() else np.nan
    work_res = np.array([rel_mean(r_, 6) for r_ in range(n_a)]); deg_res = np.array([rel_mean(r_, 7) for r_ in range(n_a)])
    with open(out / 'student_articles.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['edge_id', 'student', 'advisor', 'thesis_year', 'kci_id', 'year', 'dt', 'bin', 'institution', 'moved', 'adv_resid', 'adv_pre_resid', 'own_thesis_cos', 'work_colleague_resid', 'degree_uni_colleague_resid', 'title'])
        for r_, d in enumerate(docs):
            e = edges[d['edge_i']]
            w.writerow([e['edge_id'], e['student'], pinfo[e['professor_id']]['name'], e['year'], d['kci_id'], d['year'], d['dt'], BINS[bin_of[r_]][0] if bin_of[r_] >= 0 else '', d['inst'], int(moved[r_]),
                        round(float(adv_res[r_]), 5), '' if np.isnan(adv_pre_res[r_]) else round(float(adv_pre_res[r_]), 5), round(float(own_thesis[r_]), 4), '' if np.isnan(work_res[r_]) else round(float(work_res[r_]), 5), '' if np.isnan(deg_res[r_]) else round(float(deg_res[r_]), 5), d['label']])
    # ---- 구간별 평균 + 제자 부트스트랩
    stu_id = np.array([d['edge_i'] for d in docs]); uniq = np.unique(stu_id)
    def boot_mean(vals, sel):
        idx = np.flatnonzero(sel & ~np.isnan(vals))
        if len(idx) < 10: return None
        by = collections.defaultdict(list)
        for q in idx: by[stu_id[q]].append(vals[q])
        keys = list(by); m = float(np.mean(vals[idx])); bs = []
        for _ in range(a.boot):
            pick = rng.integers(0, len(keys), len(keys)); bs.append(float(np.mean(np.concatenate([by[keys[p]] for p in pick]))))
        return {'mean': m, 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n_articles': int(len(idx)), 'n_students': len(keys)}
    by_bin = {}
    for b, (lab, _, _) in enumerate(BINS):
        sel = bin_of == b
        by_bin[lab] = {'advisor_resid': boot_mean(adv_res, sel), 'advisor_pre_resid': boot_mean(adv_pre_res, sel), 'own_thesis_cos': boot_mean(own_thesis, sel),
                       'work_colleague_resid': boot_mean(work_res, sel), 'degree_uni_colleague_resid(moved)': boot_mean(deg_res, sel & moved),
                       'work_colleague_resid(moved)': boot_mean(work_res, sel & moved), 'work_colleague_resid(stayed)': boot_mean(work_res, sel & ~moved & (np.array([bool(d['inst']) for d in docs])))}
    # ---- 제자별 기울기(학위 후 논문 ≥4편, 연도 범위 ≥4년)
    slopes = []
    for s in uniq:
        idx = np.flatnonzero((stu_id == s) & (dts >= 0))
        if len(idx) >= 4 and (dts[idx].max() - dts[idx].min()) >= 4:
            slopes.append(float(np.polyfit(dts[idx], adv_res[idx], 1)[0]))
    slopes = np.array(slopes); slope_boot = [float(np.mean(rng.choice(slopes, len(slopes)))) for _ in range(a.boot)] if len(slopes) else []
    # ---- 회귀: 잔차 ~ 관계×구간 (학위 후만), 양방향 클러스터(제자, 교수)
    post = np.flatnonzero(bin_of >= 1); names = []; cols = []
    Rk = K[post]; Bk = bin_of[post]
    for k in (1, 3, 6, 7):
        for b in range(1, len(BINS)):
            col = ((Rk == k) & (Bk[:, None] == b)).astype(float).reshape(-1)
            if col.sum() >= 20: names.append(f'{REL[k]}×{BINS[b][0]}'); cols.append(col)
    for k in (2, 4, 5):
        col = (Rk == k).astype(float).reshape(-1)
        if col.sum() >= 20: names.append(REL[k]); cols.append(col)
    X = np.column_stack(cols); y = R[post].reshape(-1); mk = mask[post].reshape(-1)
    cl_s = np.repeat(stu_id[post], n_p); cl_p = np.tile(np.arange(n_p), len(post))
    X, y, cl_s, cl_p = X[mk], y[mk], cl_s[mk], cl_p[mk]
    beta, se = ols_2way(X, y, cl_s, cl_p)
    reg = {n: {'beta': float(b_), 'se_twoway': float(s_), 't': float(b_ / s_), 'n_pairs': int(X[:, k].sum())} for k, (n, b_, s_) in enumerate(zip(names, beta, se))}
    res = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'prelim': str(P0), 'n_edges': len(edges), 'n_students_with_cret': len(edges) - no_cret, 'n_students_with_articles': int(n_students), 'n_articles': n_a,
           'n_articles_post': int((dts >= 1).sum()), 'n_moved_articles': int(moved.sum()), 'by_bin': by_bin,
           'slope_per_year': {'n_students': int(len(slopes)), 'mean': float(slopes.mean()) if len(slopes) else None, 'ci95': [float(np.percentile(slope_boot, 2.5)), float(np.percentile(slope_boot, 97.5))] if slope_boot else None, 'share_negative': float((slopes < 0).mean()) if len(slopes) else None},
           'regression': reg}
    json.dump(res, open(out / 'trajectory.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # ---- 보고서
    def fm(x): return '—' if x is None else f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] ({x['n_students']}명/{x['n_articles']}편)"
    L = [f'# 제자 쪽 시계열 — 학위 뒤 본인 KCI 논문의 이탈과 이동', '', f"생성 {res['generated']}. 제자 논문(간선) {len(edges)}건 중 A 등급 cretId 있음 {res['n_students_with_cret']}, 본인 KCI 논문 있음 {n_students}명, 논문 {n_a}편(학위 후 {res['n_articles_post']}편, 소속이 학위 대학과 다른 논문 {res['n_moved_articles']}편). 교수 묶음 = {P0.name}. 제자 재표집 부트스트랩 {a.boot}회.", '',
         '## 1. 학위 후 경과 연수별 (잔차 코사인; 기준 교수 평균 ≈ 0)', '', '| 구간 | 지도교수(전체 묶음) | 지도교수 학위 전 10년 창(물려받은 주제; 다른 교수도 같은 창, 이원 잔차) | 본인 박사논문과의 코사인(원값) | 재직 대학 동료(계보 밖) |', '|---|---|---|---|---|']
    for lab, v in by_bin.items(): L.append(f"| {lab} | {fm(v['advisor_resid'])} | {fm(v['advisor_pre_resid'])} | {fm(v['own_thesis_cos'])} | {fm(v['work_colleague_resid'])} |")
    L += ['', '## 2. 학위 대학에 남은 사람 vs 옮긴 사람 (재직 대학 = 논문 발표 당시 소속)', '', '| 구간 | 남음: 재직(=학위) 대학 동료 | 옮김: 재직 대학 동료 | 옮김: 학위 대학 동료 |', '|---|---|---|---|']
    for lab, v in by_bin.items(): L.append(f"| {lab} | {fm(v['work_colleague_resid(stayed)'])} | {fm(v['work_colleague_resid(moved)'])} | {fm(v['degree_uni_colleague_resid(moved)'])} |")
    sl = res['slope_per_year']
    L += ['', f"## 3. 제자별 기울기 — 지도교수 잔차의 연간 변화(학위 후 논문 4편 이상·4년 이상, {sl['n_students']}명): 평균 {sl['mean']:+.5f}/년" + (f" [{sl['ci95'][0]:+.5f}, {sl['ci95'][1]:+.5f}]" if sl['ci95'] else '') + f", 음(멀어짐) 비율 {sl['share_negative']:.2f}" if sl['mean'] is not None else '## 3. 기울기 — 자료 부족', '',
          '## 4. 회귀 (학위 후 논문만; 잔차 ~ 관계×구간, 기준 = 그 외 교수; 제자·교수 양방향 클러스터 SE)', '', '| 항 | 계수 | SE | t | 쌍 |', '|---|---|---|---|---|']
    for n, v in reg.items(): L.append(f"| {n} | {v['beta']:+.4f} | {v['se_twoway']:.4f} | {v['t']:.1f} | {v['n_pairs']} |")
    L += ['', '## 읽는 법', '- 지도교수 잔차가 구간이 지날수록 줄면 이탈, 그대로면 지속. "학위 전 10년 창" 열은 지도교수가 그 뒤 주제를 바꿔도 물려받은 주제 자체에서 얼마나 멀어지는지를 본다.',
          '- 재직 대학 동료 잔차가 구간이 지날수록 커지면 재직 대학 쪽 사회화(대학 효과가 경력 뒤에 나타나는 경우). 옮긴 사람의 "학위 대학 동료" 열이 "재직 대학 동료" 열보다 크면 출신 학파를 유지하는 것.',
          '- 수치는 코사인 잔차(앞 분석과 같은 척도: 지도교수 효과 +0.026 이 기준).']
    (out / '제자궤적_보고서.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({lab: {k: (round(v[k]['mean'], 4) if v[k] else None) for k in ('advisor_resid', 'advisor_pre_resid', 'own_thesis_cos', 'work_colleague_resid')} for lab, v in by_bin.items()}, ensure_ascii=False))
    print('slope', sl); print(json.dumps({n: (round(v['beta'], 4), round(v['t'], 1)) for n, v in reg.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
