"""예비 분석 — 지도교수 연구 묶음과 제자 박사논문의 주제 유사도 (2026-10-29)

교수 묶음(문서) = 국중 전거(KAC, 등급 A)의 저작 제목(단행본·학술지 논문·학위논문, `works/nlk_works.jsonl`)
                + KCI 논문 제목·키워드(저자 이름 = 교수 이름 이고 발표 당시 소속에 교수 소속 대학이 있는 것, ART*.xml author-group)
제자 쪽 = 학맥 간선의 박사논문 제목(+ thesis-topics 인물·개념어 태그)
표현 = bge-m3 제목 임베딩(로컬 ONNX GPU) 코사인, 그리고 온톨로지 어휘(인물·개념어) 태그 Jaccard
기준선 = 같은 대학 다른 교수 / 전체 교수 / 대학×10년 코호트 안 순열(1,000회)
출력(--out): prof_docs.csv, edges_similarity.csv, professors_summary.csv, universities_summary.csv, university_distance.csv,
            decade_summary.csv, permutation.json, 예비분석_보고서.md
실행: <ocr_train venv python> -X utf8 prelim_similarity.py --repo <<SITE>> --base <KCI_현대문학_2000-2026> --out <폴더>
2026-11-04: --prof-source nlk_toc|all 로 국중 단행본 목차(nlk_toc_fetch.py, 책 한 권 = 문서 하나 = 제목 + 장 제목, h2k, 600자)를 교수 묶음에 더한다. --prof-subset 으로 표본 고정.
이름만 같은 사람을 교수와 자동으로 합치지 않는다 — KCI 논문은 이름+소속 대학이 모두 맞을 때만 쓴다.
"""
import argparse, csv, json, re, sys, glob, html, unicodedata, collections, random, math, time
from pathlib import Path
import numpy as np

def nm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·]+', '', s)

def uni_key(s):
    s = unicodedata.normalize('NFKC', s or '').replace('여자대학교', '여대').replace('대학교', '대')
    m = re.search(r'([가-힣A-Za-z]+?(?:여대|대))(?![학])', s)
    return m.group(1) if m else ''

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--min-docs', type=int, default=5, help='교수 묶음 최소 문서 수')
    ap.add_argument('--perm', type=int, default=1000); ap.add_argument('--no-embed', action='store_true'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cret-ids', help='match_cret_ids.py 의 cret_best.csv — 주면 KCI 논문은 이름·소속 대신 A 등급 cretId 집합(cret_ids_A)으로만 모은다(2026-10-29 「A만 먼저」)')
    ap.add_argument('--prof-source', choices=['both', 'kci', 'nlk', 'nlk_toc', 'all'], default='both',
                    help='교수 묶음 출처: 국중 저작+KCI(기본) / KCI 논문만 / 국중 저작만 / nlk_toc = 국중 저작 제목 + 단행본 목차(KCI 없음) / all = KCI(초록) + 국중 저작 제목 + 목차 (2026-11-04)')
    ap.add_argument('--toc-dir', help='nlk_toc_fetch.py 출력 폴더(<kac>.jsonl, toc_lines). 기본 <base>/참고문헌_구축/저자전거/works/toc')
    ap.add_argument('--toc-chars', type=int, default=600, help='한 책의 목차를 문서에 붙일 때 최대 글자 수(초록 600자와 같은 상한)')
    ap.add_argument('--prof-subset', help='professor_id 열이 있는 CSV(예: 공개 파생표 professors.csv) — 주면 이 교수들로만 묶음을 쓴다(표본 고정 비교용)')
    ap.add_argument('--abstract', action='store_true', help='KCI 논문 문서에 국문 초록(source_articles.csv abstract_ko, 앞 600자)을 붙여 임베딩 — prof_docs.csv 에는 제목만 적는다')
    a = ap.parse_args()
    repo, base, out = Path(a.repo), Path(a.base), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed)
    sys.path.insert(0, str(base / '수집도구')); sys.path.insert(0, str(repo))
    from kci_ontology import Matcher
    try:
        from hanja_norm import h2k
    except Exception:
        h2k = lambda s, name=False: s
    matcher = Matcher(base / '수집도구' / 'ontology_lexicon.json')
    def tags(text):
        return {(h['kind'], h['id']) for h in matcher.match(text, '') if not h.get('ambiguity_review')}

    # ---- 학맥
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs = {n['id']: n for n in gl['nodes'] if n.get('kind') == 'professor'}
    edges = [e for e in gl['edges'] if e.get('source') in profs and e.get('title')]
    tp = json.load(open(repo / 'site/data/thesis-topics-public.json', encoding='utf-8'))
    ttags = {t['id']: t for t in tp['topics']}
    rp = json.load(open(repo / 'site/data/research-public.json', encoding='utf-8'))
    th_by_id = {t['id']: t for t in rp['theses']}

    # ---- 교수 ↔ KAC (등급 A)
    kac_rows = list(csv.DictReader(open(base / '참고문헌_구축/저자전거/sparql/전거연결후보.csv', encoding='utf-8-sig', newline='')))
    prof_kacs = collections.defaultdict(list)
    for r in kac_rows:
        if r['grade'] != 'A' or '교수' not in r['our_roles']: continue
        for pid in [x.strip() for x in r['our_ids'].split(';')]:
            if pid in profs: prof_kacs[pid].append(r['kac'])
    works = {}
    wpath = base / '참고문헌_구축/저자전거/works/nlk_works.jsonl'
    if wpath.exists():
        for line in open(wpath, encoding='utf-8'):
            rec = json.loads(line); works[rec['kac']] = rec['works']
    # ---- 단행본 목차 (nlk_toc_fetch.py): 저작 URI → 장 제목 줄 (쪽수 '= 12' 는 뗀다). 목차 본문은 비공개 — prof_docs.csv 에는 제목만 적는다
    use_toc = a.prof_source in ('nlk_toc', 'all'); toc_of = {}
    if use_toc:
        tdir = Path(a.toc_dir) if a.toc_dir else base / '참고문헌_구축/저자전거/works/toc'
        for f in sorted(glob.glob(str(tdir / 'KAC*.jsonl'))):
            for line in open(f, encoding='utf-8'):
                rec = json.loads(line)
                lines = [re.sub(r'\s*=\s*\d+\s*$', '', x).strip() for x in rec.get('toc_lines') or []]
                lines = [x for x in lines if x]
                if lines: toc_of[rec['uri']] = lines
        print('books with toc', len(toc_of), 'from', tdir, flush=True)
    subset = None
    if a.prof_subset:
        subset = {r['professor_id'] for r in csv.DictReader(open(a.prof_subset, encoding='utf-8-sig', newline='')) if r.get('professor_id')}
        print('professor subset', len(subset), flush=True)

    # ---- KCI 논문: 이름 + 소속 대학 일치
    kci = {}
    for r in csv.DictReader(open(base / '서지정보/전체수집목록.csv', encoding='utf-8-sig', newline='')):
        kci[r['kci_id']] = r
    abstracts = {}
    if a.abstract:
        for r in csv.DictReader(open(base / '참고문헌_API/source_articles.csv', encoding='utf-8-sig', newline='')):
            if r.get('abstract_ko', '').strip(): abstracts[r['kci_id']] = re.sub(r'\s+', ' ', r['abstract_ko']).strip()[:600]
        print('abstracts', len(abstracts), flush=True)
    cret_A = None; cret_arts = collections.defaultdict(set)
    if a.cret_ids:
        cret_A = {}
        for r in csv.DictReader(open(a.cret_ids, encoding='utf-8-sig', newline='')):
            if r['kind'] == '교수' and r['grade'] == 'A' and r['person_id'] in profs:
                cret_A[r['person_id']] = [x for x in (r.get('cret_ids_A') or r['cret_id']).split('; ') if x]
        for r in csv.DictReader(open(base / '참고문헌_구축/저자ID/author_ids.csv', encoding='utf-8-sig', newline='')):
            if r['cret_id']: cret_arts[r['cret_id']].add(r['kci_id'])
        print('A-grade professors with cretId', len(cret_A), 'cret clusters', len(cret_arts), flush=True)
    auth_inst = {}
    for f in glob.glob(str(base / '참고문헌_API' / 'ART*.xml')):
        t = open(f, encoding='utf-8', errors='replace').read()
        aid = Path(f).stem
        lst = []
        for m in re.finditer(r'<author\b[^>]*>(.*?)</author>', t, re.S):
            blk = m.group(1)
            nmm = re.search(r'<name>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</name>', blk, re.S)
            inst = re.search(r'<institution>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</institution>', blk, re.S)
            if nmm: lst.append((nm(nmm.group(1)), (inst.group(1).strip() if inst else '')))
        auth_inst[aid] = lst
    print('KCI articles with parsed authors', len(auth_inst), flush=True)
    by_name = collections.defaultdict(list)
    for aid, lst in auth_inst.items():
        for n, inst in lst: by_name[n].append((aid, inst))

    # ---- 교수 문서 묶음
    prof_docs = {}  # pid -> list of dict(text, year, src)
    for pid, p in profs.items():
        docs = []
        unis = {uni_key(x) for x in (p.get('affiliations') or [])} - {''}
        seen_titles = set()
        for kac in (prof_kacs.get(pid, []) if a.prof_source != 'kci' else []):
            for w in works.get(kac, []):
                title = (w.get('title') or '').strip()
                if not title or 'Sound' in w.get('types', []): continue
                key = nm(h2k(title))
                if key in seen_titles: continue
                seen_titles.add(key)
                label = h2k(title + ((' : ' + w['rest']) if w.get('rest') else ''))
                kind = 'thesis' if 'Thesis' in w.get('types', []) else 'journal' if 'ElectronicJournal' in w.get('types', []) or 'Article' in w.get('types', []) else 'book'
                text = label
                if use_toc and kind == 'book' and w.get('uri') in toc_of:
                    toc_txt = h2k(' / '.join(toc_of[w['uri']]))[:a.toc_chars]
                    text = label + '\n' + toc_txt; kind = 'book+toc'
                docs.append({'text': text, 'year': (w.get('year') or '')[:4], 'src': 'nlk:' + kind, 'label': label})
        kci_items = []
        if cret_A is not None:   # A 등급 cretId 집합으로만 (이름·소속 대조 안 함)
            for cid in cret_A.get(pid, []):
                for aid in sorted(cret_arts.get(cid, ())): kci_items.append((aid, 'cret ' + cid))
        else:
            for aid, inst in by_name.get(nm(p['name']), []):
                if not unis or not any(u and u in inst.replace('여자대학교', '여대').replace('대학교', '대') for u in unis): continue
                kci_items.append((aid, inst))
        for aid, inst in (kci_items if a.prof_source not in ('nlk', 'nlk_toc') else []):
            r = kci.get(aid)
            if not r: continue
            key = nm(r['title'])
            if key in seen_titles: continue
            seen_titles.add(key)
            kw = (r.get('keywords_english') or '').replace(';', ', ')
            text = r['title'] + ((' — ' + kw) if kw else '')
            if a.abstract and aid in abstracts: text += '\n' + abstracts[aid]
            docs.append({'text': text, 'year': r.get('year', ''), 'src': 'kci:' + inst, 'label': r['title']})
        prof_docs[pid] = docs
    with open(out / 'prof_docs.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['professor_id', 'name', 'affiliations', 'kac', 'n_docs', 'src', 'year', 'text'])
        for pid, docs in prof_docs.items():
            for d in docs: w.writerow([pid, profs[pid]['name'], ';'.join(profs[pid].get('affiliations') or []), ';'.join(prof_kacs.get(pid, [])), len(docs), d['src'], d['year'], d.get('label', d['text'])])
    usable = {pid for pid, d in prof_docs.items() if len(d) >= a.min_docs}
    if subset is not None: usable &= subset
    print('professors', len(profs), 'with KAC', len(prof_kacs), 'usable(>=%d docs)' % a.min_docs, len(usable), flush=True)

    # ---- 제자 문서
    stu_docs = []
    for e in edges:
        if e['source'] not in usable: continue
        tt = ttags.get(e.get('thesis_id'), {})
        extra = ' '.join((tt.get('people') or []) + (tt.get('concepts') or []))
        stu_docs.append({'edge': e, 'text': h2k(e['title']) + ((' — ' + extra) if extra else ''), 'title_only': h2k(e['title'])})
    print('edges usable', len(stu_docs), flush=True)

    # ---- 임베딩
    texts = []
    for pid in sorted(usable):
        for d in prof_docs[pid]: texts.append(d['text'])
    for s in stu_docs: texts.append(s['text'])
    vecs = None
    if not a.no_embed:
        import torch  # noqa
        sys.path.insert(0, r'<HOME>\Downloads\drive-download-20260918T092441Z-1-002\작업도구\추천')
        from embed_local_bge import LocalBGE
        model = LocalBGE(max_len=512 if (a.abstract or use_toc) else 128, threads=4)
        t0 = time.time(); vecs = []
        for i in range(0, len(texts), 256):
            vecs.append(np.asarray(model.encode(texts[i:i + 256], batch=64), dtype=np.float32))
        vecs = np.vstack(vecs); vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        print('embedded', vecs.shape, f'{time.time()-t0:.0f}s', flush=True)
    k = 0; prof_vec = {}; prof_vec_by_year = {}; prof_tags = {}
    for pid in sorted(usable):
        n = len(prof_docs[pid])
        if vecs is not None:
            v = vecs[k:k + n]; c = v.mean(axis=0); prof_vec[pid] = c / (np.linalg.norm(c) + 1e-9)
            prof_vec_by_year[pid] = [(d['year'], v[j]) for j, d in enumerate(prof_docs[pid])]
        prof_tags[pid] = set().union(*[tags(d['text']) for d in prof_docs[pid]]) if prof_docs[pid] else set()
        k += n
    stu_vec = vecs[k:] if vecs is not None else None

    def cos(u, v): return float(np.dot(u, v))
    def window_vec(pid, year, lo=-10, hi=5):
        try: y = int(year)
        except Exception: return None
        vs = [v for yy, v in prof_vec_by_year[pid] if yy.isdigit() and lo <= int(yy) - y <= hi]
        if len(vs) < 3: return None
        c = np.mean(vs, axis=0); return c / (np.linalg.norm(c) + 1e-9)
    def jacc(A, B): return len(A & B) / len(A | B) if (A | B) else float('nan')

    # ---- 간선별 유사도와 기준선
    pids = sorted(usable); P = np.vstack([prof_vec[p] for p in pids]) if vecs is not None else None
    prof_uni = {p: ({uni_key(x) for x in (profs[p].get('affiliations') or [])} - {''}) for p in pids}
    rows = []
    for i, s in enumerate(stu_docs):
        e = s['edge']; pid = e['source']; uni = uni_key(e.get('university', ''))
        row = {'edge_id': e['id'], 'professor_id': pid, 'professor': profs[pid]['name'], 'student': e.get('target', ''), 'thesis_id': e.get('thesis_id', ''), 'year': e.get('year', ''),
               'university': uni, 'title': e['title'], 'n_prof_docs': len(prof_docs[pid]), 'identity': (e.get('student_identity') or {}).get('status', '')}
        stags = tags(s['text']); row['n_student_tags'] = len(stags); row['tag_jaccard'] = round(jacc(stags, prof_tags[pid]), 4) if stags else ''
        row['tag_overlap'] = ' | '.join(sorted(t[1] for t in (stags & prof_tags[pid])))[:200]
        if vecs is not None:
            sv = stu_vec[i]; sims = P @ sv
            j = pids.index(pid); row['emb_sim'] = round(float(sims[j]), 4)
            row['emb_rank_all'] = int((sims > sims[j]).sum()) + 1; row['n_prof_all'] = len(pids)
            same = [kk for kk, p in enumerate(pids) if uni and uni in prof_uni[p]]
            if len(same) >= 2 and j in same:
                ss = sims[same]; row['emb_rank_same_uni'] = int((ss > sims[j]).sum()) + 1; row['n_prof_same_uni'] = len(same)
                others = [x for kk, x in zip(same, ss) if kk != j]
                row['emb_z_same_uni'] = round((sims[j] - float(np.mean(others))) / (float(np.std(others)) + 1e-9), 3)
            wv = window_vec(pid, e.get('year'))
            row['emb_sim_window'] = round(cos(wv, sv), 4) if wv is not None else ''
            row['emb_mean_other_profs'] = round(float(np.delete(sims, j).mean()), 4)
        rows.append(row)
    cols = ['edge_id', 'professor_id', 'professor', 'student', 'thesis_id', 'year', 'university', 'title', 'identity', 'n_prof_docs', 'n_student_tags', 'tag_jaccard', 'tag_overlap',
            'emb_sim', 'emb_sim_window', 'emb_mean_other_profs', 'emb_rank_all', 'n_prof_all', 'emb_rank_same_uni', 'n_prof_same_uni', 'emb_z_same_uni']
    with open(out / 'edges_similarity.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in cols})

    if vecs is not None:
        # 문서 벡터도 저장(시계열 검증용: 교수 문서 하나하나의 벡터·연도·교수 번호) — stat_review.py 가 읽는다
        dv = []; dp = []; dy = []
        k2 = 0
        for jj, pid in enumerate(pids):
            n = len(prof_docs[pid])
            for d in prof_docs[pid]:
                dp.append(jj); dy.append(int(d['year'][:4]) if str(d['year'])[:4].isdigit() else -1)
            dv.append(vecs[k2:k2 + n]); k2 += n
        np.savez_compressed(out / 'vectors.npz', stu=stu_vec, prof=P, pids=np.array(pids), edge_ids=np.array([s['edge']['id'] for s in stu_docs]),
                            prof_ndocs=np.array([len(prof_docs[p]) for p in pids]),
                            docs_vec=np.vstack(dv).astype(np.float16), docs_pid=np.array(dp, dtype=np.int32), docs_year=np.array(dy, dtype=np.int32))
        json.dump({'edges': [{'edge_id': s['edge']['id'], 'professor_id': s['edge']['source'], 'student': s['edge'].get('target', ''), 'thesis_id': s['edge'].get('thesis_id', ''),
                              'year': s['edge'].get('year', ''), 'university': uni_key(s['edge'].get('university', '')), 'identity': (s['edge'].get('student_identity') or {}).get('status', '')} for s in stu_docs],
                   'professors': {p: {'name': profs[p]['name'], 'universities': sorted(prof_uni[p]), 'n_docs': len(prof_docs[p])} for p in pids}},
                  open(out / 'vectors_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
    # ---- 순열 검정(대학×10년 코호트 안에서 지도교수 재배정)
    perm = {}
    if vecs is not None and rows:
        obs = np.array([r['emb_sim'] for r in rows]); groups = collections.defaultdict(list)
        for i, r in enumerate(rows): groups[(r['university'], (str(r['year'])[:3] + '0') if str(r['year'])[:3].isdigit() else '')].append(i)
        idx_p = np.array([pids.index(r['professor_id']) for r in rows]); S = stu_vec @ P.T  # (edges × profs)
        means = []
        for _ in range(a.perm):
            shuffled = idx_p.copy()
            for g in groups.values():
                if len(g) > 1:
                    gg = np.array(g); shuffled[gg] = idx_p[np.random.permutation(gg)]
            means.append(float(S[np.arange(len(rows)), shuffled].mean()))
        means = np.array(means)
        perm = {'observed_mean_sim': float(obs.mean()), 'perm_mean': float(means.mean()), 'perm_sd': float(means.std()), 'z': float((obs.mean() - means.mean()) / (means.std() + 1e-9)),
                'p_value_one_sided': float((means >= obs.mean()).mean()), 'n_perm': a.perm, 'n_edges': len(rows), 'groups': len(groups),
                'top1_same_uni_rate': float(np.mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rows if r.get('n_prof_same_uni')])) if any(r.get('n_prof_same_uni') for r in rows) else None,
                'chance_top1_same_uni': float(np.mean([1 / r['n_prof_same_uni'] for r in rows if r.get('n_prof_same_uni')])) if any(r.get('n_prof_same_uni') for r in rows) else None}
        # 대학별 순열
        perm['by_university'] = {}
        for uni in sorted({r['university'] for r in rows if r['university']}):
            ii = [i for i, r in enumerate(rows) if r['university'] == uni]
            if len(ii) < 10: continue
            o = obs[ii].mean(); ms = []
            sub_groups = [g for g in groups.values() if rows[g[0]]['university'] == uni]
            for _ in range(min(a.perm, 500)):
                sh = idx_p.copy()
                for g in sub_groups:
                    if len(g) > 1: gg = np.array(g); sh[gg] = idx_p[np.random.permutation(gg)]
                ms.append(float(S[np.array(ii), sh[np.array(ii)]].mean()))
            ms = np.array(ms); perm['by_university'][uni] = {'n': len(ii), 'observed': float(o), 'perm_mean': float(ms.mean()), 'z': float((o - ms.mean()) / (ms.std() + 1e-9)), 'p': float((ms >= o).mean())}
    (out / 'permutation.json').write_text(json.dumps(perm, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---- 교수·대학·연대 요약
    def mean(xs): xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]; return round(sum(xs) / len(xs), 4) if xs else ''
    by_prof = collections.defaultdict(list)
    for r in rows: by_prof[r['professor_id']].append(r)
    with open(out / 'professors_summary.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['professor_id', 'name', 'affiliations', 'n_docs', 'n_nlk', 'n_kci', 'n_advisees', 'mean_emb_sim', 'mean_emb_sim_window', 'mean_other_profs', 'top1_same_uni_rate', 'mean_tag_jaccard', 'years'])
        for pid in sorted(by_prof, key=lambda p: -len(by_prof[p])):
            rs = by_prof[pid]; d = prof_docs[pid]
            w.writerow([pid, profs[pid]['name'], ';'.join(profs[pid].get('affiliations') or []), len(d), sum(1 for x in d if x['src'].startswith('nlk')), sum(1 for x in d if x['src'].startswith('kci')), len(rs),
                        mean([r.get('emb_sim') for r in rs]), mean([r.get('emb_sim_window') for r in rs if r.get('emb_sim_window') != '']), mean([r.get('emb_mean_other_profs') for r in rs]),
                        mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')]), mean([r['tag_jaccard'] for r in rs if r['tag_jaccard'] != '']),
                        f"{min(r['year'] for r in rs)}–{max(r['year'] for r in rs)}"])
    by_uni = collections.defaultdict(list)
    for r in rows: by_uni[r['university']].append(r)
    uni_vec = {}
    with open(out / 'universities_summary.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['university', 'n_edges', 'n_professors', 'mean_emb_sim', 'mean_other_profs', 'top1_same_uni_rate', 'chance_top1', 'perm_z', 'perm_p', 'mean_tag_jaccard'])
        for uni in sorted(by_uni, key=lambda u: -len(by_uni[u])):
            rs = by_uni[uni]; pu = perm.get('by_university', {}).get(uni, {})
            w.writerow([uni, len(rs), len({r['professor_id'] for r in rs}), mean([r.get('emb_sim') for r in rs]), mean([r.get('emb_mean_other_profs') for r in rs]),
                        mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')]), mean([1 / r['n_prof_same_uni'] for r in rs if r.get('n_prof_same_uni')]),
                        round(pu.get('z', float('nan')), 3) if pu else '', pu.get('p', '') if pu else '', mean([r['tag_jaccard'] for r in rs if r['tag_jaccard'] != ''])])
            if vecs is not None and len(rs) >= 10:
                vs = [prof_vec[r['professor_id']] for r in rs]; c = np.mean(vs, axis=0); uni_vec[uni] = c / (np.linalg.norm(c) + 1e-9)
    if uni_vec:
        unis = sorted(uni_vec, key=lambda u: -len(by_uni[u]))[:10]
        with open(out / 'university_distance.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow([''] + unis)
            for u in unis: w.writerow([u] + [round(cos(uni_vec[u], uni_vec[v]), 3) for v in unis])
    by_dec = collections.defaultdict(list)
    for r in rows: by_dec[str(r['year'])[:3] + '0년대' if str(r['year'])[:3].isdigit() else '미상'].append(r)
    with open(out / 'decade_summary.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['decade', 'n_edges', 'mean_emb_sim', 'mean_emb_sim_window', 'mean_other_profs', 'top1_same_uni_rate', 'mean_tag_jaccard'])
        for d in sorted(by_dec):
            rs = by_dec[d]; w.writerow([d, len(rs), mean([r.get('emb_sim') for r in rs]), mean([r.get('emb_sim_window') for r in rs if r.get('emb_sim_window') != '']), mean([r.get('emb_mean_other_profs') for r in rs]),
                                        mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')]), mean([r['tag_jaccard'] for r in rs if r['tag_jaccard'] != ''])])
    # ---- 보고서
    L = ['# 예비 분석 보고 — 지도교수 연구 묶음과 제자 박사논문의 주제 유사도', '', f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 자료: 학맥 간선 {len(edges)}건 중 교수 묶음(문서 {a.min_docs}편 이상, 전거 A급)이 있는 {len(rows)}건. 교수 {len(by_prof)}명.', '',
         '## 전체', f"- 관측 평균 코사인(제자 논문 ↔ 지도교수 묶음): **{perm.get('observed_mean_sim', float('nan')):.4f}**",
         f"- 같은 대학×10년 코호트 안에서 지도교수를 섞은 귀무분포 평균 {perm.get('perm_mean', float('nan')):.4f} (sd {perm.get('perm_sd', float('nan')):.4f}) → z = {perm.get('z', float('nan')):.2f}, p = {perm.get('p_value_one_sided', float('nan')):.3f}",
         f"- 제자 논문만 보고 같은 대학 교수 중 지도교수를 1위로 맞힌 비율 {perm.get('top1_same_uni_rate') or 0:.3f} (우연 {perm.get('chance_top1_same_uni') or 0:.3f})", '',
         '## 대학별 (간선 10건 이상)', '| 대학 | 간선 | 교수 | 평균 유사도 | 다른 교수 평균 | 1위 맞힘 | 우연 | 순열 z | p |', '|---|---|---|---|---|---|---|---|---|']
    for uni in sorted(by_uni, key=lambda u: -len(by_uni[u])):
        rs = by_uni[uni]
        if len(rs) < 10: continue
        pu = perm.get('by_university', {}).get(uni, {})
        L.append(f"| {uni} | {len(rs)} | {len({r['professor_id'] for r in rs})} | {mean([r.get('emb_sim') for r in rs])} | {mean([r.get('emb_mean_other_profs') for r in rs])} | {mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')])} | {mean([1 / r['n_prof_same_uni'] for r in rs if r.get('n_prof_same_uni')])} | {round(pu.get('z', float('nan')), 2) if pu else ''} | {pu.get('p', '') if pu else ''} |")
    L += ['', '## 연대별', '| 연대 | 간선 | 평균 유사도 | 지도 시점 창 | 다른 교수 평균 | 1위 맞힘 |', '|---|---|---|---|---|---|']
    for d in sorted(by_dec):
        rs = by_dec[d]; L.append(f"| {d} | {len(rs)} | {mean([r.get('emb_sim') for r in rs])} | {mean([r.get('emb_sim_window') for r in rs if r.get('emb_sim_window') != ''])} | {mean([r.get('emb_mean_other_profs') for r in rs])} | {mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')])} |")
    L += ['', '## 교수별 상위(제자 5명 이상, 평균 유사도 순)', '| 교수 | 소속 | 문서 | 제자 | 평균 유사도 | 다른 교수 평균 | 1위 맞힘 |', '|---|---|---|---|---|---|---|']
    for pid in sorted([p for p in by_prof if len(by_prof[p]) >= 5], key=lambda p: -(mean([r.get('emb_sim') for r in by_prof[p]]) or 0))[:40]:
        rs = by_prof[pid]; L.append(f"| {profs[pid]['name']} | {';'.join(profs[pid].get('affiliations') or [])[:30]} | {len(prof_docs[pid])} | {len(rs)} | {mean([r.get('emb_sim') for r in rs])} | {mean([r.get('emb_mean_other_profs') for r in rs])} | {mean([1 if r.get('emb_rank_same_uni') == 1 else 0 for r in rs if r.get('n_prof_same_uni')])} |")
    if uni_vec:
        unis = sorted(uni_vec, key=lambda u: -len(by_uni[u]))[:8]
        L += ['', '## 대학 프로필 간 코사인(상위 8교)', '| | ' + ' | '.join(unis) + ' |', '|---|' + '---|' * len(unis)]
        for u in unis: L.append(f'| {u} | ' + ' | '.join(f'{cos(uni_vec[u], uni_vec[v]):.3f}' for v in unis) + ' |')
    L += ['', '## 읽는 법과 한계',
          '- 유사도는 제목(+키워드·태그) 임베딩의 코사인이다. 짧은 제목이라 값의 절대 크기보다 **기준선(다른 교수·순열)과의 차이**를 본다.',
          '- 교수 묶음은 국중 전거 저작(단행본·논문·학위논문)과 이름+소속이 맞는 KCI 논문만 넣었다. KCI 연구자 ID 가 모이면 KCI 쪽을 전수로 바꾼다.',
          '- 지도 시점 창(−10~+5년)은 문서 연도가 있는 것만 썼다. 연도 없는 저작은 전체 묶음에만 들어간다.',
          '- 선택 효과(비슷한 학생이 비슷한 교수를 택함)와 지도 효과는 이 설계로 구분되지 않는다. 같은 대학 안 비교와 순열은 "우연보다 닮았는가"만 답한다.',
          '- 교수 식별은 전거 등급 A 만, 제자 식별은 학맥 간선 그대로(상태 미확인 포함; identity 열로 걸러 민감도 분석 가능).']
    (out / '예비분석_보고서.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps({k: v for k, v in perm.items() if k != 'by_university'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
