"""학풍 분석 공개 파생표 내보내기 (2026-10-31, 연구자 「파생표 먼저」)

논문 심사자·독자가 그림·표의 수치를 다시 계산하고 원자료를 공개 식별자로 되찾을 수 있도록,
분석에 실제로 들어간 표본·관계·유사도·토픽·결과를 CSV 로 내보낸다. **원문·초록·임베딩 벡터·토큰 캐시·로컬 경로는 넣지 않는다.**

입력(--base = KCI_현대문학_2000-2026, --repo = <SITE> 체크아웃):
  참고문헌_구축/학풍_A기준_KCI만_전체/{vectors.npz, vectors_meta.json, 검증*/stat_review.json, 제자궤적/*}
  참고문헌_구축/학풍_TFIDF_토픽/{bge_subset,tfidf,lsa300,lda50}/{vectors.npz, vectors_meta.json, 검증*/stat_review.json}, topics_k50.csv 등
  참고문헌_구축/저자전거/{인물식별표_A.csv, prof_phd_years.csv}, 저자ID/{author_ids.csv, 식별/cret_best.csv}, 서지정보/전체수집목록.csv
  site/data/genealogy-library.json
출력(--out, 기본 참고문헌_구축/공개_파생표_YYYYMMDD):
  theses.csv, professors.csv, professor_articles.csv, lineage_professor_edges.csv,
  pair_relations_similarity.csv.gz (논문×교수 쌍: 관계 유형 3단계 × 잔차·원 유사도 5표현),
  thesis_topics_k50.csv, professor_topics_k50.csv, topics_k50.csv, lineage_topics_k50.csv, university_topics_k50.csv, eta2_k50.json,
  regression_results.csv, decade_results.csv, window_results.csv, trajectory_bins.csv, trajectory_regression.csv, student_articles.csv,
  methods.json, README.md
검사: 모든 출력 파일에서 로컬 경로 조각(드라이브 문자·사용자 폴더·동기화 폴더 이름)이 발견되면 assert 로 멈춘다.
실행: py -X utf8 export_derived_tables.py --base <KCI_현대문학_2000-2026> --repo <<SITE>> [--out <폴더>] [--as-of 2026-10-31]
"""
import argparse, csv, json, gzip, re, html, unicodedata, collections, shutil, time
from pathlib import Path
import numpy as np

REL = ['advisor', 'grand', 'sibling', 'uncle', 'lineage_other', 'same_uni_outside_lineage', 'same_uni_x_lineage']
LAB = {'advisor': '지도교수', 'grand': '조부 교수', 'sibling': '형제 교수', 'uncle': '삼촌 교수', 'lineage_other': '같은 계보 기타', 'same_uni_outside_lineage': '계보 밖 같은 대학', 'same_uni_x_lineage': '같은 대학×계보', 'base': '기준(다른 대학·계보 밖)'}
TIERS = [('all', ''), ('confirmed', '확정'), ('confirmed_inferred', '확정,추론')]
TIER_DIR = {'all': '검증', 'confirmed': '검증_확정', 'confirmed_inferred': '검증_확정추론'}
TIER_LAB = {'all': '전체 간선(확정+추론+미확인)', 'confirmed': '확정 간선만', 'confirmed_inferred': '확정+추론 간선'}
HEX = re.compile(r'^[0-9a-f]{16,32}$')


def nm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·]+', '', s)


def resid2(S):
    return S - S.mean(axis=1, keepdims=True) - S.mean(axis=0, keepdims=True) + S.mean()


def riss_url(tid):
    return f'https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no={tid}' if HEX.match(tid or '') else ''


def lineage_struct(gl, profs_all, allow_status):
    """stat_review.py 와 같은 규칙: 교수→교수 간선(상태 제한) → advisor_of(첫 간선 유지), students_of, root."""
    advisor_of = {}; used = 0; total = 0
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']:
            total += 1; st = (e.get('student_identity') or {}).get('status')
            if allow_status and st not in allow_status: continue
            used += 1; advisor_of.setdefault(e['target'], e['source'])
    students_of = collections.defaultdict(set)
    for s, t in advisor_of.items(): students_of[t].add(s)
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
        return cur
    return advisor_of, students_of, root, used, total


def relation_matrix(edges, pids, pinfo, advisor_of, students_of, root):
    """stat_review.py 의 R 행렬과 같은 코드(0 기준, 1..5 계보, 6 계보 밖 같은 대학, 9..12 = 계보(2..5)+같은 대학)."""
    pidx = {p: i for i, p in enumerate(pids)}; prof_unis = [set(pinfo[p]['universities']) for p in pids]
    roots = {p: root(p) for p in set(pids) | set(advisor_of) | set(advisor_of.values())}; root_size = collections.Counter(roots[p] for p in pids)
    R = np.zeros((len(edges), len(pids)), dtype=np.int8)
    for i, e in enumerate(edges):
        a_ = e['professor_id']; u = e['university']; g = advisor_of.get(a_); sibs = students_of.get(a_, set()); uncles = (students_of.get(g, set()) - {a_}) if g else set(); r = roots.get(a_)
        for j, pj in enumerate(pids):
            same = bool(u) and (u in prof_unis[j])
            if pj == a_: k = 1
            elif g and pj == g: k = 2
            elif pj in sibs: k = 3
            elif pj in uncles: k = 4
            elif root_size.get(r, 0) >= 2 and roots.get(pj) == r: k = 5
            elif same: k = 6
            else: k = 0
            if k in (2, 3, 4, 5) and same: k = 7 + k
            R[i, j] = k
    return R, roots


def rel_label(k):
    if k == 0: return 'base', 0
    if k >= 9: return REL[k - 8], 1
    return REL[k - 1], 1 if k == 6 else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out'); ap.add_argument('--as-of', default=time.strftime('%Y-%m-%d'))
    ap.add_argument('--extra', default='', help='추가 표현: 이름=폴더(vectors.npz·vectors_meta.json·검증*/stat_review.json) 를 쉼표로 — 예 ko_sroberta=<…>/학풍_임베딩다양화_20261101/ko_sroberta')
    ap.add_argument('--review-root', help='정정판 결과 폴더: <표현>/검증*/stat_review.json, 제자궤적/, 토픽/ 을 여기서 읽는다(벡터는 원 분석 폴더 그대로)')
    ap.add_argument('--reuse-tables', help='이전 공개 파생표 폴더: professor_articles.csv 와 professors.csv 의 식별자 열을 그대로 쓴다(저자 식별 파일이 옛 판으로 되돌아간 경우)')
    a = ap.parse_args(); base, repo = Path(a.base), Path(a.repo)
    RV = Path(a.review_root) if a.review_root else None
    def rv_dir(v, d):
        return (RV / {'bge': 'bge_m3'}.get(v, v)) if RV else d
    out = Path(a.out) if a.out else base / '참고문헌_구축' / ('공개_파생표_' + a.as_of.replace('-', '')); out.mkdir(parents=True, exist_ok=True)
    G = base / '참고문헌_구축'; MAIN = G / '학풍_A기준_KCI만_전체'; LEX = G / '학풍_TFIDF_토픽'
    t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:5.0f}s]', *x, flush=True)
    def wcsv(name, cols, rows):
        with open(out / name, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        log(name, len(rows), '행')

    # ---------- 학맥·표본
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    nodes = {x['id']: x for x in gl['nodes']}
    z = np.load(MAIN / 'vectors.npz', allow_pickle=True); meta = json.load(open(MAIN / 'vectors_meta.json', encoding='utf-8'))
    edges, pinfo, pids = meta['edges'], meta['professors'], list(z['pids']); n_e, n_p = len(edges), len(pids)
    assert [e['edge_id'] for e in edges] == list(z['edge_ids'])
    struct = {}
    for tier, st in TIERS:
        allow = {x for x in st.split(',') if x}
        advisor_of, students_of, root, used, total = lineage_struct(gl, profs_all, allow)
        R, roots = relation_matrix(edges, pids, pinfo, advisor_of, students_of, root)
        struct[tier] = {'advisor_of': advisor_of, 'roots': roots, 'R': R, 'used': used, 'total': total}
        log('관계 행렬', tier, f'교수→교수 간선 {used}/{total}', collections.Counter(int(x) for x in R.ravel()))
    # 전체 간선 기준의 R 은 stat_review.json 의 쌍 수와 맞아야 한다
    sr_main = json.load(open(rv_dir('bge', MAIN) / '검증/stat_review.json', encoding='utf-8'))
    Rall = struct['all']['R']
    for k, n in enumerate(REL[:5], start=1):
        got = int(((Rall == k) | (Rall == 7 + k)).sum()) if k >= 2 else int((Rall == 1).sum())
        if n in sr_main['main']: assert got == sr_main['main'][n]['n_pairs'], (n, got, sr_main['main'][n]['n_pairs'])
    assert int((Rall == 6).sum()) == sr_main['main']['same_uni_outside_lineage']['n_pairs']
    log('관계 행렬이 stat_review.json 쌍 수와 일치')

    # ---------- 어휘 표현 표본(1,177)과 벡터
    variants = {'bge': MAIN}
    for v in ['bge_subset', 'tfidf', 'lsa300', 'lda50']: variants[v] = LEX / v
    for kv in [x for x in a.extra.split(',') if x.strip()]:
        k, _, vpath = kv.partition('='); variants[k.strip()] = Path(vpath.strip())
    sims = {}; lex_edge_ids = None
    for v, d in variants.items():
        zz = np.load(d / 'vectors.npz', allow_pickle=True); mm = json.load(open(d / 'vectors_meta.json', encoding='utf-8'))
        assert list(zz['pids']) == pids, v
        eids = list(zz['edge_ids']); S = zz['stu'].astype(np.float64) @ zz['prof'].astype(np.float64).T; Rz = resid2(S)
        sims[v] = {'edge_ids': eids, 'raw': S, 'resid': Rz, 'note': mm.get('note', ''), 'dim': int(zz['stu'].shape[1])}
        if v == 'lda50': sims[v]['theta_stu'] = zz['stu'].astype(np.float64) ** 2; sims[v]['theta_prof'] = zz['prof'].astype(np.float64) ** 2
        full = (eids == list(z['edge_ids']))   # bge 와 같은 간선 전체(새 임베딩 모델)인지, 어휘 표본(1,177)인지
        sims[v]['full_sample'] = full
        if not full:
            lex_edge_ids = eids if lex_edge_ids is None else lex_edge_ids; assert eids == lex_edge_ids, v
        log('유사도', v, S.shape, '잔차 SD', round(float(Rz.std()), 4))
    assert abs(float(sims['bge']['resid'].std()) - sr_main['sd_resid']) < 1e-6
    lex_set = set(lex_edge_ids)

    # ---------- 보조 표
    ida = {r['person_id']: r for r in csv.DictReader(open(G / '저자전거/인물식별표_A.csv', encoding='utf-8-sig', newline=''))}
    phd = {r['professor_id']: r for r in csv.DictReader(open(G / '저자전거/prof_phd_years.csv', encoding='utf-8-sig', newline=''))}
    cret_best = {r['person_id']: r for r in csv.DictReader(open(G / '저자ID/식별/cret_best.csv', encoding='utf-8-sig', newline=''))}
    cret_arts = collections.defaultdict(set)
    for r in csv.DictReader(open(G / '저자ID/author_ids.csv', encoding='utf-8-sig', newline='')):
        if r['cret_id']: cret_arts[r['cret_id']].add(r['kci_id'])
    kci = {}
    for r in csv.DictReader(open(base / '서지정보/전체수집목록.csv', encoding='utf-8-sig', newline='')):
        kci[r['kci_id']] = {'title': r['title'], 'year': r['year'], 'journal': r['journal'], 'selection': r['selection']}
    log('보조 표', len(ida), len(phd), len(cret_best), len(cret_arts), len(kci))

    # ---------- theses.csv
    th_rows = []; gl_by_id = {x['id']: x for x in gl['edges']}
    for i, e in enumerate(edges):
        ge = gl_by_id.get(e['edge_id'])
        st_node = nodes.get(e['student'], {})
        row = {'edge_id': e['edge_id'], 'thesis_id': e['thesis_id'], 'riss_url': riss_url(e['thesis_id']), 'student_node_id': e['student'], 'student_name': st_node.get('name', ''),
               'title': (ge or {}).get('title', ''), 'year': e['year'], 'university': e['university'], 'advisor_professor_id': e['professor_id'], 'advisor_name': pinfo[e['professor_id']]['name'],
               'advisor_name_raw_in_catalog': (ge or {}).get('advisor_raw', ''), 'advisor_basis': (ge or {}).get('basis', ''), 'student_identity_status': e['identity'],
               'student_identity_status_raw': ((ge or {}).get('student_identity') or {}).get('status_raw', ''), 'in_bge_sample': 1, 'in_lexical_sample': int(e['edge_id'] in lex_set)}
        for tier, _ in TIERS: row[f'lineage_root_{tier}'] = struct[tier]['roots'].get(e['professor_id'], e['professor_id'])
        th_rows.append(row)
    wcsv('theses.csv', list(th_rows[0].keys()), th_rows)

    # ---------- professors.csv + professor_articles.csv
    pr_rows = []; art_rows = []; mismatch = []
    for j, p in enumerate(pids):
        node = profs_all.get(p, {}); ia = ida.get(p, {}); ph = phd.get(p, {}); cb = cret_best.get(p, {})
        cret_A = [x for x in (cb.get('cret_ids_A') or cb.get('cret_id') or '').split('; ') if x] if cb.get('grade') == 'A' else []
        row = {'professor_id': p, 'name': pinfo[p]['name'], 'universities': ';'.join(pinfo[p]['universities']), 'affiliations_genealogy': ';'.join(node.get('affiliations', [])),
               'identity_status': node.get('identity_status', ''), 'identity_status_raw': node.get('identity_status_raw', ''), 'own_thesis_ids': ';'.join(node.get('thesis_ids', [])),
               'phd_year': ph.get('phd_year', '') if not str(ph.get('source', '')).startswith(('ambiguous', 'none')) else '', 'phd_year_source': ph.get('source', ''), 'phd_university': ph.get('university', ''),
               # 동명 인물이 같은 KAC 를 받은 행은 연구자 판정 전까지 어느 쪽에도 주지 않는다(인물식별표 규칙) → 공개표에서는 비우고 표지만 남긴다
               'kac': '' if ia.get('kac_shared_with') else ia.get('kac', ''), 'kac_evidence': '보류(동명 KAC 공유)' if ia.get('kac_shared_with') else ia.get('kac_evidence', ''),
               'isni': '' if ia.get('kac_shared_with') else ia.get('isni', ''), 'kac_shared_with_namesake': ia.get('kac_shared_with', ''),
               'cret_ids_A': '; '.join(cret_A), 'n_cret_ids_A': len(cret_A), 'cret_rule': cb.get('rule', ''), 'n_kci_articles_in_bundle': int(pinfo[p]['n_docs']),
               'n_advisee_theses_in_sample': sum(1 for e in edges if e['professor_id'] == p)}
        for tier, _ in TIERS:
            row[f'advisor_professor_id_{tier}'] = struct[tier]['advisor_of'].get(p, ''); row[f'lineage_root_{tier}'] = struct[tier]['roots'].get(p, p)
        pr_rows.append(row)
        # 묶음 재구성: A 등급 cretId 의 KCI 논문, 제목 정규화 중복 제거(prelim_similarity.py 와 같은 규칙), 목록에 있는 논문만
        seen = set(); n = 0
        for cid in cret_A:
            for aid in sorted(cret_arts.get(cid, ())):
                r = kci.get(aid)
                if not r: continue
                key = nm(r['title'])
                if key in seen: continue
                seen.add(key); n += 1
                art_rows.append({'professor_id': p, 'kci_id': aid, 'kci_url': f'https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId={aid}', 'cret_id': cid, 'year': r['year'], 'journal': r['journal'], 'title': r['title'], 'selection': r['selection']})
        if n != int(pinfo[p]['n_docs']): mismatch.append((p, n, int(pinfo[p]['n_docs'])))
    if a.reuse_tables:
        OLD = Path(a.reuse_tables)
        oldp = {r['professor_id']: r for r in csv.DictReader(open(OLD / 'professors.csv', encoding='utf-8-sig', newline=''))}
        gl_year = {}
        for e in gl['edges']:
            if e.get('target') in profs_all and str(e.get('year', '')).isdigit(): gl_year.setdefault(e['target'], (e['year'], e.get('university', '')))
        n_fix = 0
        for row in pr_rows:
            o = oldp[row['professor_id']]
            for c in ('kac', 'kac_evidence', 'isni', 'kac_shared_with_namesake', 'cret_ids_A', 'n_cret_ids_A', 'cret_rule', 'phd_year', 'phd_year_source', 'phd_university'): row[c] = o[c]
            assert str(row['n_kci_articles_in_bundle']) == str(o['n_kci_articles_in_bundle']), row['professor_id']
            if row['own_thesis_ids'] != o['own_thesis_ids']:   # 본인 논문 연결이 정정된 교수: 옛 학맥 간선에서 온 학위 연도는 버리고 현행 간선에서 다시 읽는다
                n_fix += 1
                if row['professor_id'] in gl_year: row['phd_year'], row['phd_university'] = gl_year[row['professor_id']]; row['phd_year_source'] = 'genealogy_edge'
                elif o['phd_year_source'] == 'genealogy_edge': row['phd_year'] = ''; row['phd_university'] = ''; row['phd_year_source'] = 'cleared_after_own_thesis_correction'
        art_rows = list(csv.DictReader(open(OLD / 'professor_articles.csv', encoding='utf-8-sig', newline='')))
        cnt_art = collections.Counter(r['professor_id'] for r in art_rows)
        assert all(cnt_art[p] == int(pinfo[p]['n_docs']) for p in pids), '묶음 논문 수 불일치'
        log('이전 파생표 재사용: professor_articles', len(art_rows), '· 식별자 열 · 본인 논문 연결이 바뀐 교수', n_fix)
    wcsv('professors.csv', list(pr_rows[0].keys()), pr_rows)
    wcsv('professor_articles.csv', list(art_rows[0].keys()), art_rows)
    if not a.reuse_tables: log('묶음 재구성 불일치 교수', len(mismatch), mismatch[:10])

    # ---------- lineage_professor_edges.csv
    le_rows = []
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']:
            si = e.get('student_identity') or {}
            le_rows.append({'edge_id': e['id'], 'advisor_professor_id': e['source'], 'advisor_name': profs_all[e['source']]['name'], 'student_professor_id': e['target'], 'student_name': profs_all[e['target']]['name'],
                            'thesis_id': e['thesis_id'], 'riss_url': riss_url(e['thesis_id']), 'title': e.get('title', ''), 'year': e.get('year', ''), 'university': e.get('university', ''),
                            'identity_status': si.get('status', ''), 'identity_status_raw': si.get('status_raw', ''), 'basis': e.get('basis', ''),
                            'advisor_in_analysis_sample': int(e['source'] in set(pids)), 'student_in_analysis_sample': int(e['target'] in set(pids))})
    wcsv('lineage_professor_edges.csv', list(le_rows[0].keys()), le_rows)

    # ---------- pair_relations_similarity.csv.gz
    lex_idx = {eid: i for i, eid in enumerate(lex_edge_ids)}
    cols = ['edge_id', 'professor_id', 'same_university'] + [f'relation_{t}' for t, _ in TIERS] + [f'same_uni_x_lineage_{t}' for t, _ in TIERS] + [f'resid_{v}' for v in variants] + [f'raw_{v}' for v in variants]
    n_rows = 0
    with gzip.open(out / 'pair_relations_similarity.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i, e in enumerate(edges):
            li = lex_idx.get(e['edge_id']); u = e['university']
            for j, p in enumerate(pids):
                rec = [e['edge_id'], p, int(bool(u) and u in pinfo[p]['universities'])]
                rels = [rel_label(int(struct[t]['R'][i, j])) for t, _ in TIERS]
                rec += [r for r, _ in rels] + [x if r not in ('base', 'same_uni_outside_lineage', 'advisor') else 0 for r, x in rels]
                for key in ('resid', 'raw'):
                    for v in variants:
                        if sims[v]['full_sample']: rec.append(f"{sims[v][key][i, j]:.5f}")
                        else: rec.append(f"{sims[v][key][li, j]:.5f}" if li is not None else '')
                w.writerow(rec); n_rows += 1
    log('pair_relations_similarity.csv.gz', n_rows, '행')

    # ---------- 토픽 θ
    th = sims['lda50']
    rows = [{'edge_id': eid, 'thesis_id': eid.split('|')[-1], **{f't{k}': f'{th["theta_stu"][i, k]:.5f}' for k in range(50)}} for i, eid in enumerate(lex_edge_ids)]
    wcsv('thesis_topics_k50.csv', list(rows[0].keys()), rows)
    rows = [{'professor_id': p, 'name': pinfo[p]['name'], **{f't{k}': f'{th["theta_prof"][j, k]:.5f}' for k in range(50)}} for j, p in enumerate(pids)]
    wcsv('professor_topics_k50.csv', list(rows[0].keys()), rows)
    for fn in ['topics_k50.csv', 'lineage_topics_k50.csv', 'university_topics_k50.csv', 'eta2_k50.json']:
        srcf = (RV / '토픽' / fn) if (RV and (RV / '토픽' / fn).exists()) else (LEX / fn)
        shutil.copyfile(srcf, out / fn); log('복사', fn, '(정정판)' if srcf != LEX / fn else '')

    # ---------- 회귀 결과표 (5 표현 × 3 간선 단계)
    reg_rows = []; dec_rows = []; win_rows = []
    for v, d in variants.items():
        for tier, _ in TIERS:
            p = rv_dir(v, d) / TIER_DIR[tier] / 'stat_review.json'
            if not p.exists(): log('없음', p.name, v, tier); continue
            sr = json.load(open(p, encoding='utf-8'))
            ppe = sr.get('prof_prof_edges') or {}
            for n, m in sr['main'].items():
                reg_rows.append({'representation': v, 'edge_tier': tier, 'edge_tier_label': TIER_LAB[tier], 'prof_prof_edges_used': ppe.get('used', ''), 'prof_prof_edges_all': ppe.get('all', ''),
                                 'n_theses': sr['n_theses'], 'n_professors': sr['n_professors'], 'sd_resid': f"{sr['sd_resid']:.6f}", 'relation': n, 'relation_label': LAB[n],
                                 'beta': f"{m['beta']:.6f}", 'se_thesis': f"{m['se_thesis']:.6f}", 'se_prof': f"{m['se_prof']:.6f}", 'se_twoway': f"{m['se_twoway']:.6f}", 't_twoway': f"{m['t_twoway']:.3f}",
                                 'p_twoway': f"{m['p_twoway']:.3g}", 'p_holm': f"{m['p_holm']:.3g}", 'perm_p': '' if m.get('perm_p') is None else f"{m['perm_p']:.3f}", 'perm_null_sd': '' if m.get('perm_null_sd') is None else f"{m['perm_null_sd']:.6f}",
                                 'ci95_boot_prof_lo': f"{m['ci95_boot_prof'][0]:.6f}", 'ci95_boot_prof_hi': f"{m['ci95_boot_prof'][1]:.6f}", 'ci95_boot_thesis_lo': f"{m['ci95_boot_thesis'][0]:.6f}", 'ci95_boot_thesis_hi': f"{m['ci95_boot_thesis'][1]:.6f}",
                                 'std_effect': f"{m['std_effect']:.4f}", 'raw_cos_mean': f"{m['raw_cos_mean']:.6f}", 'raw_cos_diff_vs_base': f"{m['raw_cos_diff_vs_base']:.6f}", 'n_pairs': m['n_pairs'], 'boot': sr['boot'], 'perm': sr['perm']})
            for lab, dd in sr.get('decade', {}).items():
                for n, m in dd.items():
                    if n == 'n_theses': continue
                    dec_rows.append({'representation': v, 'edge_tier': tier, 'decade': lab, 'n_theses': dd['n_theses'], 'relation': n, 'relation_label': LAB[n], 'beta': f"{m['beta']:.6f}", 'se_twoway': f"{m['se_twoway']:.6f}", 't_twoway': f"{m['t']:.3f}"})
            for lab, ww in sr.get('windows', {}).items():
                for n, m in ww.items():
                    ci = m.get('ci95_thesis_boot')
                    win_rows.append({'representation': v, 'edge_tier': tier, 'window': lab, 'relation': n, 'relation_label': LAB.get(n, n), 'mean_resid': '' if m['mean_resid'] is None else f"{m['mean_resid']:.6f}", 'n_pairs': m['n_pairs'],
                                     'n_theses': m.get('n_theses', ''), 'ci95_thesis_boot_lo': f'{ci[0]:.6f}' if ci else '', 'ci95_thesis_boot_hi': f'{ci[1]:.6f}' if ci else ''})
    wcsv('regression_results.csv', list(reg_rows[0].keys()), reg_rows); wcsv('decade_results.csv', list(dec_rows[0].keys()), dec_rows); wcsv('window_results.csv', list(win_rows[0].keys()), win_rows)

    # ---------- 제자 궤적
    TJ = (RV / '제자궤적') if RV else (MAIN / '제자궤적')
    tj = json.load(open(TJ / 'trajectory.json', encoding='utf-8'))
    tb = []
    for lab, d in tj['by_bin'].items():
        for k, m in d.items():
            if not isinstance(m, dict): continue
            tb.append({'bin': lab, 'measure': k, 'mean': f"{m['mean']:.6f}" if m.get('mean') is not None else '', 'ci95_lo': f"{m['ci95'][0]:.6f}" if m.get('ci95') else '', 'ci95_hi': f"{m['ci95'][1]:.6f}" if m.get('ci95') else '', 'n_articles': m.get('n_articles', ''), 'n_students': m.get('n_students', '')})
    wcsv('trajectory_bins.csv', list(tb[0].keys()), tb)
    tr = [{'term': n, 'beta': f"{m['beta']:.6f}", 'se_twoway': f"{m['se_twoway']:.6f}", 't_twoway': f"{m['t']:.3f}", 'n_pairs': m['n_pairs']} for n, m in tj['regression'].items()]
    sl = tj['slope_per_year']; tr.append({'term': 'slope_per_year(advisor_resid, per student)', 'beta': f"{sl['mean']:.6f}", 'se_twoway': '', 't_twoway': '', 'n_pairs': sl['n_students']})
    wcsv('trajectory_regression.csv', list(tr[0].keys()), tr)
    sa_cols = ['edge_id', 'student', 'advisor', 'thesis_year', 'kci_id', 'year', 'dt', 'bin', 'institution', 'moved', 'adv_resid', 'adv_pre_resid', 'own_thesis_cos', 'work_colleague_resid', 'degree_uni_colleague_resid', 'title']
    sa = list(csv.DictReader(open(TJ / 'student_articles.csv', encoding='utf-8-sig', newline='')))
    assert set(sa[0].keys()) == set(sa_cols), sa[0].keys()
    wcsv('student_articles.csv', sa_cols, sa)

    # ---------- methods.json
    methods = {
        'as_of': a.as_of, 'generated': time.strftime('%Y-%m-%d %H:%M'),
        'sample': {'theses_bge': n_e, 'theses_lexical': len(lex_edge_ids), 'professors': n_p, 'professor_kci_articles': int(sum(pinfo[p]['n_docs'] for p in pids)), 'pairs_bge': n_e * n_p, 'pairs_lexical': len(lex_edge_ids) * n_p,
                   'genealogy_as_of': gl.get('as_of'), 'prof_prof_edges': {t: {'used': struct[t]['used'], 'all': struct[t]['total']} for t, _ in TIERS},
                   'student_identity_status_in_sample': dict(collections.Counter(e['identity'] for e in edges))},
        'data_sources': {
            'theses': 'RISS 통합목록 박사학위논문(control_no = thesis_id, 16~32자 16진수), 서울대 Primo(snu:), 통합목록 미대응(DDMKL_UNMATCHED_)',
            'genealogy': '<SITE> site/data/genealogy-library.json (교수·학위자 노드, 지도 간선, 동일인 판정 3단계 확정/추론/미확인)',
            'professor_articles': 'KCI Open API articleDetail 수집본(국어국문학·문학 학술지 124종 82,344편, 2000–2026) 중 교수의 A 등급 KCI 연구자 ID(cretId) 집합에 속한 논문. cretId 는 KCI 논문 페이지 저자 링크에 공개된 식별자',
            'authority': '국립중앙도서관 LOD 인물 전거(KAC)·ISNI — A 등급(문학 분야 + 소속/논문 제목 일치)만. 동명 인물이 같은 KAC 를 받은 교수는 kac·isni 를 비우고 kac_shared_with_namesake 에 상대를 적음(연구자 판정 전)',
            'embeddings_not_included': 'bge-m3 임베딩·TF-IDF·LSA·LDA θ 의 원 벡터, KCI 초록, 박사논문 본문·토큰 캐시는 포함하지 않음(원문 복원 가능성). 쌍별 코사인·잔차와 토픽 θ(박사논문·교수 묶음)만 공개'},
        'representations': {v: {'dim': sims[v]['dim'], 'note': sims[v]['note'] or ('bge-m3 (BAAI/bge-m3, max_len 512) — 제자 = 박사논문 제목+개념어·인물 태그, 교수 묶음 = KCI 논문 제목+영문 키워드+국문 초록 600자 평균(L2)' if v == 'bge' else ''),
                                'n_theses': len(sims[v]['edge_ids']), 'resid_sd': float(sims[v]['resid'].std()), 'sample': '전체 1,334' if sims[v]['full_sample'] else '어휘 표본(본문 색인 있음)'} for v in variants},
        'lexical_pipeline': {'tokenizer': 'kiwipiepy 명사(NNG·NNP), 한자 → 한글(hanja_norm.h2k), 숫자+년대 유지, 학술 상투어 불용어', 'corpus': 'KCI 82,344편(제목+국문 초록 72,042편) + 박사논문 본문 1,593편(문단 단위 토큰화)',
                             'vocab': {'min_df': 5, 'max_df': 0.4, 'max_features': 30000}, 'tfidf': 'sublinear tf, 교수 묶음 = 논문 벡터 합(L2)', 'lsa': 'TruncatedSVD 300', 'lda': 'LatentDirichletAllocation K=50 online 10 passes seed 42; 문서 벡터 = sqrt(θ) L2 (내적 = Bhattacharyya 계수); 교수 묶음 = 논문 θ 평균의 sqrt',
                             'theta_in_tables': 'thesis_topics_k50 = stu 벡터의 제곱(θ 합 1), professor_topics_k50 = prof 벡터의 제곱(평균 θ 를 합 1 로 정규화)'},
        'model': {'dependent': '제자 박사논문 × 교수 묶음 유사도에서 논문 평균·교수 평균을 뺀 이원 고정효과 잔차(resid_*). 표본별 전체 행렬에서 잔차화(bge 1,334×324, 어휘 1,177×324)',
                  'relations': {k: LAB[k] for k in REL}, 'relation_coding': '기준 = 다른 대학·계보 밖. 계보 관계(조부·형제·삼촌·기타)이면서 같은 대학이면 same_uni_x_lineage=1 로 추가 더미(stat_review.py 와 동일)',
                  'lineage_construction': '교수→교수 지도 간선(동일인 판정 상태로 제한 가능: all / confirmed / confirmed_inferred)에서 지도교수(첫 간선)·형제(같은 지도교수의 다른 교수 제자)·조부·삼촌(조부의 다른 제자)·같은 뿌리(뿌리 크기 ≥2)를 구성',
                  'se': '논문·교수 양방향 클러스터(Cameron–Gelbach–Miller) SE, Holm 보정(관계 항 7개), 교수/논문 클러스터 부트스트랩 300회(백분위 95%), 같은 대학 안 교수 열 순열 300회(같은-대학 항에는 정보 없음), seed 42',
                  'std_effect': 'beta / 잔차 SD', 'windows': '지도교수 묶음을 학위 연도 기준 창(학위 전 10년 / 전후 3년 / 후 10년 / 위약 ±11년 이후)으로 나눠 창별 행렬을 따로 잔차화(문서 3편 이상인 교수만)'},
        'scripts': ['prelim_similarity.py (bge-m3 묶음·간선 벡터)', 'tfidf_topic_replication.py (어휘 표현·토픽)', 'stat_review.py (회귀·검정; --edge-status 로 간선 단계)', 'lineage_sensitivity.py (단계별 재실행)', 'student_trajectory.py (제자 궤적)', 'match_cret_ids.py · build_identity_table.py (A 등급 식별)', 'export_derived_tables.py (이 표)'],
        'caveats': ['수집 범위(RISS 통합목록 박사논문 + 교수 명단 + 도서관 조회) 안의 관계만 센 값이며 전체 학계가 아니다', '이름만 같은 저자는 자동 병합하지 않았다. A 등급 cretId·KAC 는 규칙 기반 판정이며 연구자 검토표가 따로 있다',
                    '학위자 동일인 판정이 미확인인 간선이 전체의 대부분이다(이 표 student_identity_status). 교수→교수 간선의 판정 상태별 재추정(edge_tier)이 regression_results 에 있다',
                    '조부(3세대) 효과는 표현과 표본 구성에 민감하다(학맥 정정 전 판에서는 확정 간선만으로 bge·tfidf 가 Holm 비유의였고, 정정판에서는 간선 대부분이 확정이어서 단계별 차이가 거의 없다)', '학맥 정정판: 2026-11-04 교수 본인 박사논문 연결 전수 재검토와 그 뒤 추가 간선을 반영한 계보로 관계 부호·회귀·제자 궤적·계보 토픽을 다시 계산했다. 표본(논문 × 교수)과 유사도는 정정 전과 같다', '같은-대학 항의 순열 p 는 같은 대학 안에서 섞으므로 정보가 없다(p=1)']}
    json.dump(methods, open(out / 'methods.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---------- README.md
    L = [f'# 학풍(지도교수 계보) 분석 공개 파생표 — {a.as_of}', '',
         '한국 현대문학 박사학위논문과 지도교수의 KCI 논문 묶음 사이의 주제 유사도를, 지도교수·계보·대학 관계별로 비교한 분석의 재현 자료다. 원문·초록·임베딩 벡터는 넣지 않고, 쌍별 유사도(원·잔차)와 토픽 비중, 관계 표지, 결과표, 그리고 원자료를 되찾을 공개 식별자(RISS control_no, KCI ART id·cretId, 국중 KAC·ISNI)만 담았다.', '',
         f"표본: 박사논문 {n_e}편(어휘 표현은 본문 색인이 있는 {len(lex_edge_ids)}편) × 교수 {n_p}명(KCI 논문 묶음 {methods['sample']['professor_kci_articles']}편). 학맥 기준일 {gl.get('as_of')}. 교수→교수 간선 {struct['all']['total']}건(확정 {struct['confirmed']['used']} · 확정+추론 {struct['confirmed_inferred']['used']}).", '',
         '## 파일', '', '| 파일 | 행 | 내용 |', '|---|---|---|',
         f'| theses.csv | {n_e} | 제자 박사논문(간선 1건 = 논문 1편): thesis_id(RISS control_no → riss_url), 제목·연도·학위 대학(정규화 표기), 지도교수 id·이름, 통합목록의 지도교수 원표기, 학위자 동일인 판정(확정/추론/미확인 + 원 판정), 어휘 표본 포함 여부, 간선 단계별 계보 뿌리 |',
         f'| professors.csv | {n_p} | 교수: 이름·소속(분석용 정규화·학맥 원표기), 동일인 판정, 본인 박사논문 id, 박사학위 연도(출처: 학맥 간선 / 국중 박사논문 KAC / 이름+학과 유일), KAC·ISNI(A 등급), A 등급 cretId 집합과 판정 규칙, 묶음 논문 수, 표본 안 지도 논문 수, 간선 단계별 지도교수·뿌리 |',
         f'| professor_articles.csv | {len(art_rows)} | 교수 묶음에 들어간 KCI 논문: kci_id(→ kci_url)·cretId·연도·학술지·제목·현대문학 선별 상태. 제목 정규화 중복 제거 뒤 묶음(prelim_similarity.py 규칙) |',
         f'| lineage_professor_edges.csv | {len(le_rows)} | 교수→교수 지도 간선(계보 구성의 원자료): 지도교수·제자 교수·논문·연도·대학·동일인 판정 3단계와 원 판정·근거·표본 포함 여부 |',
         f'| pair_relations_similarity.csv.gz | {n_rows} | 논문 × 교수 쌍 전체. same_university, 간선 단계별 관계(relation_all / relation_confirmed / relation_confirmed_inferred ∈ base·advisor·grand·sibling·uncle·lineage_other·same_uni_outside_lineage)와 같은 대학×계보 표지, 표현별 잔차 유사도(resid_<표현>)와 원 유사도(raw_<표현>; lda 는 Bhattacharyya 계수). 표현 = ' + ', '.join(variants) + '. 어휘 표본(bge_subset·tfidf·lsa300·lda50) 밖 논문은 그 열이 빈칸 |',
         f'| thesis_topics_k50.csv | {len(lex_edge_ids)} | 박사논문 본문의 LDA(K=50) 토픽 비중 θ |', f'| professor_topics_k50.csv | {n_p} | 교수 묶음의 토픽 비중(논문 θ 평균, 합 1) |',
         '| topics_k50.csv | 50 | 토픽별 상위 낱말 20개와 KCI 전체·현대문학·박사논문 비중 |', '| lineage_topics_k50.csv / university_topics_k50.csv | — | 계보 뿌리별·학위 대학별 토픽 비중과 특이 토픽 |', '| eta2_k50.json | — | sqrt θ 공간에서 지도교수·계보·대학 묶음의 η² 와 순열 귀무 평균 |',
         f'| regression_results.csv | {len(reg_rows)} | 표현 5 × 간선 단계 3 의 관계별 계수·SE(논문/교수/양방향)·t·p·Holm p·순열 p·부트스트랩 구간·표준화 효과·쌍 수 |',
         f'| decade_results.csv | {len(dec_rows)} | 학위 연대별(≤2004 / 2005–2014 / 2015–) 관계 계수 |', f'| window_results.csv | {len(win_rows)} | 지도교수 묶음의 학위 연도 기준 창별 평균 잔차(전승 방향) |',
         f'| trajectory_bins.csv / trajectory_regression.csv | {len(tb)} / {len(tr)} | 제자 본인의 KCI 논문이 학위 뒤 지도교수·형제·재직 대학 동료 묶음과 얼마나 닮는지(학위 후 연차 구간별) |',
         f'| student_articles.csv | {len(sa)} | 제자 궤적의 논문별 자료: kci_id·연도·학위 후 연차·발표 당시 소속·이동 여부·잔차 유사도(초록 없음) |',
         '| methods.json | — | 표본·출처·표현·모형·스크립트·주의 사항 |', '',
         '## 관계 부호', '', '| 코드 | 뜻 |', '|---|---|'] + [f'| {k} | {v} |' for k, v in LAB.items()] + ['',
         '같은 대학×계보: 계보 관계(조부·형제·삼촌·기타)이면서 제자의 학위 대학이 그 교수의 소속이면 1. 회귀에서는 계보 더미와 함께 켜지는 추가 더미다.', '',
         '## 간선 단계(edge_tier)', '', '교수→교수 지도 간선의 학위자 동일인 판정으로 계보를 제한한 민감도 분석. `all` = 확정+추론+미확인 전체(본 분석), `confirmed` = 확정만, `confirmed_inferred` = 확정+추론. 지도교수→논문 간선은 단계와 무관하므로 지도교수 항은 세 단계에서 같다.', '',
         '## 재계산 요령', '', '- 주 결과(`regression_results.csv` 의 beta): `pair_relations_similarity.csv.gz` 에서 표현별 `resid_*` 를 종속변수로, `relation_<tier>` 더미 6개 + `same_uni_x_lineage_<tier>` 를 설명변수로 OLS(상수항 없음, 기준 = base). 어휘 표현은 어휘 열이 비어 있지 않은 행만.',
         '- 양방향 클러스터 SE 는 edge_id(논문)와 professor_id(교수) 두 클러스터 분산의 합에서 HC0 를 뺀 것.', '- 토픽 η²: `thesis_topics_k50.csv` 의 sqrt θ 를 `theses.csv` 의 지도교수·계보 뿌리·대학으로 묶어 집단 간 제곱합 비율.',
         '- KAC·ISNI 가 빈 교수 중 `kac_shared_with_namesake` 가 있는 행은 동명 인물과 전거를 공유해 판정을 보류한 것이다(교수 묶음은 KAC 가 아니라 cretId 로 만들었으므로 분석에는 영향이 없다).', '- 원자료 되찾기: 박사논문은 riss_url, KCI 논문은 kci_url(초록·참고문헌은 KCI 포털·Open API), 인물 전거는 KAC(`https://lod.nl.go.kr/resource/{KAC}`)·ISNI.', '',
         '## 주의', ''] + [f'- {c}' for c in methods['caveats']] + ['', '이 표의 수치는 수집 자료 안의 연결만 센 값이다. 이름이 같은 저자를 자동으로 합치지 않았으며, 동일인 판정은 3단계로 표시했다.']
    if RV: L += ['', '## 학맥 정정판', '', f"이 판은 학맥 기준일 {gl.get('as_of')} 의 정정된 계보로 관계 부호·회귀·제자 궤적·계보 토픽을 다시 계산한 것이다. 표본과 유사도(원·잔차)는 2026-11-01 판과 같고, professor_articles.csv 와 professors.csv 의 식별자 열은 2026-11-01 판을 그대로 옮겼다."]
    (out / 'README.md').write_text('\n'.join(L), encoding='utf-8')

    # ---------- 유출 점검
    bad = re.compile(r'(?i)(^|[^A-Za-z])[A-Z]:\\|/c/|<ONEDRIVE>|<USER>|\\Users\\|Codex|AppData|참고문헌_구축|KCI_현대문학_2000-2026|수집도구')
    for p in sorted(out.iterdir()):
        txt = gzip.open(p, 'rt', encoding='utf-8').read() if p.suffix == '.gz' else p.read_text(encoding='utf-8-sig')
        hits = [m.group(0) for m in bad.finditer(txt)]
        assert not hits, (p.name, hits[:5])
        assert 'abstract' not in txt.split('\n', 1)[0].lower(), p.name
    log('유출 점검 통과', len(list(out.iterdir())), '파일 →', out.name)


if __name__ == '__main__':
    main()
