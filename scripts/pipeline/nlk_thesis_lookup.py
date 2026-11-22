"""국립중앙도서관 LOD SPARQL 에서 학위자·교수 이름의 **박사학위논문** 레코드를 받아 우리 박사논문 목록과 맞춘다.

NLK 학위논문 레코드(bibo:Thesis, bibo:degree <…/degrees/phd>)에는 dc:creator(표기), dcterms:creator(KAC 전거 → ISNI),
nlon:degreeYear, dc:publisher(대학), nlon:department, dcterms:description("지도교수: ○○○" — 2015년 박사논문의 49%)이 있다(2026-10-29 확인).

1차: 이름(dc:creator 리터럴)으로 조회. 2차(--kac-from): 이름 조회(nlk_sparql_lookup) 결과의 문학·교수 후보 KAC 로 dcterms:creator 조회
      (옛 논문은 저자 표기가 한자라 이름 리터럴로는 안 잡히는 것을 보완).
출력(--out):
  nlk_phd_theses.jsonl / .csv   찾은 박사논문 전부(이름·KAC 기준), 지도교수 노트 파싱(advisor_note)
  thesis_matches.csv            우리 박사논문(research-public.json theses) ↔ NLK 레코드 대응: 연도·대학·제목 비교 등급
  progress.json, errors.jsonl, raw/ 는 두지 않는다(응답이 작다)
등급: A 연도(±1)·대학·제목 모두 맞음 / A* 논문은 같은데 저자 표기가 다름(개명·필명·음역·오기 — 검토) / B 연도·제목 맞고 대학 불명 또는 대학·제목 맞고 연도 ±2 / C 제목만 비슷(0.6 이상) / X 같은 이름 박사논문이 있으나 안 맞음
"""
import argparse, csv, json, os, re, sys, time, html, unicodedata, datetime as dt, difflib
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

EP = 'https://lod.nl.go.kr/sparql'
UA = '<SITE>-research/1.0 (academic genealogy study)'
PFX = ('PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX bibo: <http://purl.org/ontology/bibo/> '
       'PREFIX nlon: <http://lod.nl.go.kr/ontology/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ')
PHD = '<http://purl.org/ontology/bibo/degrees/phd>'
SEL = ('SELECT ?w ?title ?rest ?alt ?year ?issued ?pub ?kac ?dept ?desc ?creator WHERE { %s a bibo:Thesis ; bibo:degree ' + PHD + ' . '
       'OPTIONAL { ?w dcterms:title ?title } OPTIONAL { ?w nlon:remainderOfTitle ?rest } OPTIONAL { ?w dcterms:alternative ?alt } '
       'OPTIONAL { ?w nlon:degreeYear ?year } OPTIONAL { ?w dcterms:issued ?issued } OPTIONAL { ?w dc:publisher ?pub } OPTIONAL { ?w dcterms:creator ?kac } '
       'OPTIONAL { ?w nlon:department ?dept } OPTIONAL { ?w dcterms:description ?desc } OPTIONAL { ?w dc:creator ?creator } } LIMIT 100 OFFSET %d')

def nm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·]+', '', s)

def tnorm(s):
    s = unicodedata.normalize('NFKC', s or '')
    return re.sub(r'[\s\W_]+', '', s).casefold()

def q(query, timeout):
    url = EP + '?' + urllib.parse.urlencode({'query': query, 'format': 'application/sparql-results+json'})
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': 'application/sparql-results+json', 'User-Agent': UA}), timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))['results']['bindings']

def fetch_theses(subject_pattern, timeout, interval):
    rows = []; off = 0
    while True:
        b = q(PFX + SEL % (subject_pattern, off), timeout); rows += b
        if len(b) < 100: break
        off += 100; time.sleep(interval)
    theses = {}
    for r in rows:
        w = r['w']['value']
        t = theses.setdefault(w, {'kdm': w.rsplit('/', 1)[-1], 'uri': w, 'title': '', 'rest': '', 'alt': set(), 'year': '', 'issued': '', 'pub': '', 'kac': set(), 'dept': set(), 'desc': set(), 'creator': set()})
        for k in ('title', 'rest', 'year', 'issued', 'pub'):
            if k in r: t[k] = r[k]['value']
        for k in ('alt', 'dept', 'desc', 'creator'):
            if k in r: t[k].add(r[k]['value'])
        if 'kac' in r: t['kac'].add(r['kac']['value'].rsplit('/', 1)[-1])
    out = []
    for t in theses.values():
        for k in ('alt', 'kac', 'dept', 'desc', 'creator'): t[k] = sorted(t[k])
        adv = []
        for d in t['desc']:
            m = re.search(r'(?:지도교수|指導敎授|指導教授|Adviser|Advisor)\s*[:：]\s*(.+)', d)
            if m: adv.append(m.group(1).strip().rstrip('.'))
        t['advisor_note'] = ' ; '.join(adv)
        out.append(t)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--kac-from', help='nlk_sparql_matches.jsonl — 문학·교수 후보 KAC 로 2차 조회')
    ap.add_argument('--titles', action='store_true', help='우리 박사논문 주제목(원표기·한글)으로도 조회(저자 표기가 한자·로마자라 이름으로 안 잡히는 논문 보완)')
    ap.add_argument('--interval', type=float, default=0.3); ap.add_argument('--timeout', type=float, default=120); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    try:
        from hanja_norm import h2k, uni_norm
    except Exception:
        h2k = lambda s, name=False: s; uni_norm = lambda s: s
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = [r['name'] for r in csv.DictReader(open(a.names, encoding='utf-8-sig', newline=''))]
    subjects = [('name', n, '?w dc:creator %s ;' % json.dumps(n, ensure_ascii=False)) for n in names]
    if a.kac_from:
        # 2차 KAC 조회 대상 = (아직 A/B 대응이 없는 박사논문의 저자 이름 + 교수 이름)의 문학 분야 후보 전거만
        LIT = re.compile(r'국어 ?국문|한국 ?문학|한국 ?시|한국 ?소설|문학 ?평론|문학\(예술\)|비평|국어 ?교육|아동 ?문학|한국 ?수필|한국 ?희곡|문예 ?창작|현대 ?시|현대 ?소설|현대 ?문학|평론가')
        need = set()
        for r in csv.DictReader(open(a.names, encoding='utf-8-sig', newline='')):
            if 'professor' in r.get('kinds', ''): need.add(r['name'])
        prev = Path(a.out) / 'thesis_matches.csv'
        if prev.exists():
            for r in csv.DictReader(open(prev, encoding='utf-8-sig', newline='')):
                if r['grade'] not in ('A', 'B'): need.add(nm(r['our_author']))
        else:
            need.update(r['name'] for r in csv.DictReader(open(a.names, encoding='utf-8-sig', newline='')))
        seen = set()
        for line in open(a.kac_from, encoding='utf-8'):
            r = json.loads(line)
            if r['kac'] in seen or r.get('matched_name') not in need: continue
            if LIT.search(' '.join(r.get('fields', []) + r.get('jobs', []))):
                seen.add(r['kac']); subjects.append(('kac', r['kac'], '?w dcterms:creator <%s> ;' % r['kac_uri']))
        print('kac subjects', len(seen), 'for', len(need), 'names', flush=True)
    rp = json.load(open(Path(a.repo) / 'site/data/research-public.json', encoding='utf-8'))
    if a.titles:
        for th in rp['theses']:
            if not th.get('title'): continue
            mains = []
            for t in (th.get('title_raw'), th.get('title')):
                if not t: continue
                main = re.split(r'\s[:=]\s|\s:|：', unicodedata.normalize('NFKC', t))[0].strip()
                if len(main) >= 4 and main not in mains: mains.append(main)
            for i, m in enumerate(mains):
                subjects.append(('title', f"{th.get('id','')}#{i}", '?w dcterms:title %s ;' % json.dumps(m, ensure_ascii=False)))
    jl = out / 'nlk_phd_theses.jsonl'
    done = set()
    if jl.exists():
        for line in open(jl, encoding='utf-8'):
            try: done.add(tuple(json.loads(line)['query'][:2]))
            except Exception: pass
    state = 'running'; attempted = 0; errs = 0; t0 = time.time()
    def progress():
        (out / 'progress.json').write_text(json.dumps({'state': state, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'total_queries': len(subjects),
            'completed_queries': len(done), 'attempted_this_run': attempted, 'elapsed_s': round(time.time() - t0), 'pid': os.getpid()}, ensure_ascii=False, indent=2), encoding='utf-8')
    progress()
    with open(jl, 'a', encoding='utf-8') as fj, open(out / 'errors.jsonl', 'a', encoding='utf-8') as fe:
        for kind, key, pat in subjects:
            if (kind, key) in done: continue
            if (out / 'STOP').exists(): state = 'stopped'; break
            if a.limit is not None and attempted >= a.limit: state = 'batch_finished'; break
            attempted += 1
            try:
                theses = fetch_theses(pat, a.timeout, a.interval); errs = 0
            except Exception as e:
                errs += 1; fe.write(json.dumps({'query': [kind, key], 'error': type(e).__name__, 'at': dt.datetime.now(dt.timezone.utc).isoformat()}, ensure_ascii=False) + '\n'); fe.flush()
                print(kind, key, 'error', type(e).__name__, flush=True)
                if errs >= 10: state = 'stopped_on_error'; break
                time.sleep(a.interval * 10); continue
            fj.write(json.dumps({'query': [kind, key], 'n': len(theses), 'theses': theses}, ensure_ascii=False) + '\n'); fj.flush()
            done.add((kind, key))
            if attempted % 25 == 0: progress()
            time.sleep(a.interval)
        else:
            state = 'pass_finished'
    progress()

    # ---- 표로 정리 + 우리 박사논문과 대응
    all_t = {}
    for line in open(jl, encoding='utf-8'):
        rec = json.loads(line)
        for t in rec['theses']:
            e = all_t.setdefault(t['kdm'], {**t, 'found_by': set()})
            e['found_by'].add('%s:%s' % tuple(rec['query'][:2]))
    cols = ['kdm', 'found_by', 'creator', 'kac', 'title', 'rest', 'alt', 'year', 'issued', 'pub', 'pub_norm', 'dept', 'advisor_note', 'advisor_hangul', 'desc', 'uri']
    with open(out / 'nlk_phd_theses.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for t in sorted(all_t.values(), key=lambda x: x['kdm']):
            w.writerow({**{k: (' | '.join(sorted(t[k])) if isinstance(t[k], (list, set)) else t[k]) for k in cols if k in t},
                        'pub_norm': uni_norm(t['pub']), 'advisor_hangul': h2k(t['advisor_note'], name=True) if t['advisor_note'] else ''})
    by_name = {}; by_title_query = {}
    for t in all_t.values():
        for c in t['creator']: by_name.setdefault(nm(c), []).append(t)
        for c in t['creator']: by_name.setdefault(nm(h2k(c, name=True)), []).append(t)
        for fb in t['found_by']:
            if fb.startswith('title:'): by_title_query.setdefault(fb[6:].split('#')[0], []).append(t)
    mcols = ['thesis_id', 'our_author', 'our_year', 'our_university', 'our_title', 'our_advisor', 'grade', 'kdm', 'nlk_creator', 'kac', 'nlk_year', 'nlk_university', 'nlk_title', 'nlk_dept',
             'nlk_advisor_note', 'nlk_advisor_hangul', 'advisor_agree', 'title_sim', 'n_same_name_phd']
    rows = []; summary = {}
    for th in rp['theses']:
        if not th.get('author'): continue
        cands = by_name.get(nm(th['author']), []) + by_name.get(nm(h2k(th['author'], name=True)), []) + by_title_query.get(th.get('id', ''), [])
        seen = set(); cands = [c for c in cands if not (c['kdm'] in seen or seen.add(c['kdm']))]
        our_author_n = nm(h2k(th['author'], name=True))
        our_title = tnorm(h2k(th.get('title_raw') or th.get('title', '')))
        our_uni = uni_norm(th.get('university_raw') or th.get('university', ''))
        best = None
        for c in cands:
            full = h2k((c['title'] or '') + ' ' + (c['rest'] or ''))
            sim = max(difflib.SequenceMatcher(None, our_title, tnorm(full)).ratio(), difflib.SequenceMatcher(None, our_title, tnorm(h2k(c['title'] or ''))).ratio())
            yr = int(c['year']) if str(c['year']).isdigit() else (int(c['issued'][:4]) if str(c['issued'])[:4].isdigit() else None)
            oy = int(th['year']) if str(th.get('year', '')).isdigit() else None
            ydiff = abs(yr - oy) if (yr is not None and oy is not None) else None
            uni_ok = bool(our_uni) and uni_norm(c['pub']) == our_uni
            # 제목으로 찾은 레코드는 저자 표기(한자→한글 포함)도 맞아야 A/B
            author_ok = (not c['creator']) or any(nm(h2k(x, name=True)) == our_author_n or nm(x) == nm(th['author']) for x in c['creator'])
            if not author_ok and sim >= 0.9 and ydiff is not None and ydiff <= 1 and uni_ok:
                g = 'A*'  # 논문(제목·연도·대학)은 같은데 저자 표기가 다름 — 개명·필명·음역·오기 가능, 검토 대상
                cand = (g, sim, c, ydiff)
                if best is None or ('AA*BCX'.find(g), -sim) < ('AA*BCX'.find(best[0]), -best[1]): best = cand
                continue
            if not author_ok: sim = min(sim, 0.59)
            if sim >= 0.6 and ydiff is not None and ydiff <= 1 and uni_ok: g = 'A'
            elif sim >= 0.6 and ((ydiff is not None and ydiff <= 1) or (uni_ok and ydiff is not None and ydiff <= 2)): g = 'B'
            elif sim >= 0.6: g = 'C'
            else: g = 'X'
            cand = (g, sim, c, ydiff)
            if best is None or ('AA*BCX'.find(g), -sim) < ('AA*BCX'.find(best[0]), -best[1]): best = cand
        if best is None:
            rows.append({'thesis_id': th.get('id', ''), 'our_author': th['author'], 'our_year': th.get('year', ''), 'our_university': th.get('university', ''), 'our_title': th.get('title', ''),
                         'our_advisor': th.get('advisor', ''), 'grade': '—', 'n_same_name_phd': 0}); summary['—'] = summary.get('—', 0) + 1; continue
        g, sim, c, ydiff = best
        adv_h = h2k(c['advisor_note'], name=True) if c['advisor_note'] else ''
        ours_advs = [nm(h2k(x, name=True)) for x in re.split(r'[,;/·]', th.get('advisor', '') or '') if nm(x)]
        nlk_advs = [nm(h2k(x, name=True)) for x in re.split(r'[,;/·]', adv_h) if nm(x)]
        if not (ours_advs and nlk_advs): agree = ''
        elif set(ours_advs) == set(nlk_advs): agree = '예'
        elif set(ours_advs) & set(nlk_advs): agree = '일부'
        else: agree = '아니오'
        rows.append({'thesis_id': th.get('id', ''), 'our_author': th['author'], 'our_year': th.get('year', ''), 'our_university': th.get('university', ''), 'our_title': th.get('title', ''),
                     'our_advisor': th.get('advisor', ''), 'grade': g, 'kdm': c['kdm'], 'nlk_creator': ' | '.join(c['creator']), 'kac': ' | '.join(c['kac']), 'nlk_year': c['year'] or c['issued'],
                     'nlk_university': c['pub'], 'nlk_title': (c['title'] or '') + ((' : ' + c['rest']) if c['rest'] else ''), 'nlk_dept': ' | '.join(c['dept']),
                     'nlk_advisor_note': c['advisor_note'], 'nlk_advisor_hangul': adv_h, 'advisor_agree': agree, 'title_sim': f'{sim:.2f}', 'n_same_name_phd': len(cands)})
        summary[g] = summary.get(g, 0) + 1
    with open(out / 'thesis_matches.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=mcols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in mcols})
    adv_stats = {'A_star_author_spelling_differs': sum(1 for r in rows if r['grade'] == 'A*'),
                 'A_with_advisor': sum(1 for r in rows if r['grade'] == 'A' and r.get('nlk_advisor_note')),
                 'A_advisor_agree': sum(1 for r in rows if r['grade'] == 'A' and r.get('advisor_agree') == '예'),
                 'A_advisor_partial': sum(1 for r in rows if r['grade'] == 'A' and r.get('advisor_agree') == '일부'),
                 'A_advisor_disagree': sum(1 for r in rows if r['grade'] == 'A' and r.get('advisor_agree') == '아니오'),
                 'A_fills_missing_advisor': sum(1 for r in rows if r['grade'] == 'A' and r.get('nlk_advisor_note') and not r.get('our_advisor'))}
    (out / 'summary.json').write_text(json.dumps({'state': state, 'queries': len(subjects), 'nlk_phd_theses_found': len(all_t), 'our_theses': len(rows), 'grades': summary, **adv_stats}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'state': state, 'found': len(all_t), 'grades': summary, **adv_stats}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
