"""KCI 연구자 ID(cretId) ↔ 우리 인물(교수·학위자) 식별표 (2026-10-29)

입력(읽기만): 참고문헌_구축/저자ID/author_ids.csv(kci_author_ids.py 수집, 진행 중이어도 됨) · 참고문헌_API/source_articles.csv(제목·연도)
            저장소 genealogy-library.json(교수·소속) · research-public.json(학위자·학위 대학·연도·제목)
            참고문헌_구축/저자전거/sparql/전거연결후보.csv(인물→KAC, A·B급) · thesis/thesis_matches.csv(학위논문→KAC, A·A*급)
            저자전거/works/nlk_works.csv(KAC 저작 제목) · 저자전거/isni/isni_records.csv(KAC→KRI·저작 제목)
방법: 같은 이름(공백 제거)의 KCI 저자 행을 cretId 로 묶고(소속·연도·논문 제목), 우리 인물마다 cretId 후보에 근거를 매긴다.
  title   = cretId 논문 제목이 그 인물 KAC 의 국중 저작(학술지 논문 포함)·ISNI 저작 제목과 일치(정규화 완전 일치)  — 가장 강함
  inst    = cretId 소속(정규화 OO대) = 교수 소속 / 학위자의 학위 대학
  topic   = 학위논문 제목과 cretId 논문 제목의 글자 2-gram 겹침(최대) >= 0.25   — 학위자만
  early   = 학위 연도보다 10년 넘게 앞선 논문 수(학위자) — 많으면 다른 사람일 가능성
등급: A = title >= 1, 또는 inst 이고 같은 이름의 다른 cretId 에는 inst 가 없음
      B = inst 이나 다른 후보에도 inst 있음 / 후보 하나뿐 + (topic 또는 교수) / 후보 여럿이나 topic 있는 것 하나뿐
      C = 후보 하나뿐, 다른 근거 없음        D = 후보 여럿, 가를 근거 없음        — = 수집된 KCI 저자에 그 이름 없음
주의: KCI 는 한 사람에게 cretId 를 여럿 준다(소속·시기별 분리 — 같은 소속·같은 국중 저작 제목이 두 ID 에서 모두 맞는 교수가 50명+).
      그래서 인물의 KCI 식별자는 cretId 하나가 아니라 **A 등급 cretId 의 집합**(cret_ids_A)이고, 뒤 분석은 KAC(우리 인물) 단위로 묶는다.
      A 가 여럿인데 한쪽만 제목 근거가 있으면(mixed_evidence) 검토표에 올린다.
자동 병합 금지 — 등급표는 연구자 검토용. 수집이 끝나면 같은 명령으로 다시 만든다.
출력(--out): cret_candidates.csv(인물×cretId 전부) · cret_best.csv(인물별 최선) · 요약.json · 연구자_검토표_KCI저자ID.xlsx
"""
import argparse, ast, csv, json, re, sys, time, unicodedata, collections
from pathlib import Path


def add_repo(repo):
    sys.path.insert(0, str(repo))
    global h2k, uni_norm
    from hanja_norm import h2k, uni_norm


def nm(s):
    s = unicodedata.normalize('NFKC', str(s or ''))
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·\-]+', '', s)


def tnorm(s):
    s = h2k(unicodedata.normalize('NFKC', str(s or '')))
    return re.sub(r'[^0-9a-zA-Z가-힣]+', '', s).lower()


def grams(s):
    s = tnorm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def lit(v):
    if isinstance(v, list):
        return v
    try:
        return list(ast.literal_eval(v)) if v else []
    except Exception:
        return [v] if v else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); base, repo, out = Path(a.base), Path(a.repo), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    add_repo(repo)
    # ── KCI 저자 행 → cretId 묶음
    arts = {r['kci_id']: r for r in csv.DictReader(open(base / '참고문헌_API/source_articles.csv', encoding='utf-8-sig', newline=''))}
    rows = list(csv.DictReader(open(base / '참고문헌_구축/저자ID/author_ids.csv', encoding='utf-8-sig', newline='')))
    prog = json.load(open(base / '참고문헌_구축/저자ID/progress.json', encoding='utf-8'))
    clusters = collections.defaultdict(lambda: {'names': collections.Counter(), 'inst': collections.Counter(), 'years': [], 'arts': [], 'titles': []})
    skipped = collections.Counter()
    for r in rows:
        n = r['name_ko'].replace(' ', '')
        if not r['cret_id']:
            skipped['cret 없음'] += 1; continue
        if ',' in n or n == 'orcid바로가기' or not n:
            skipped['저자 묶음·orcid 행'] += 1; continue
        c = clusters[r['cret_id']]; c['names'][nm(n)] += 1
        inst = uni_norm(r['institution']) if r['institution'] else ''
        if inst:
            c['inst'][inst] += 1
        art = arts.get(r['kci_id'], {}); y = str(art.get('year', ''))[:4]
        if y.isdigit():
            c['years'].append(int(y))
        c['arts'].append(r['kci_id']); c['titles'].append(art.get('title', ''))
    by_name = collections.defaultdict(set)
    for cid, c in clusters.items():
        for n in c['names']:
            by_name[n].add(cid)
    # ── 우리 인물
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    rp = json.load(open(repo / 'site/data/research-public.json', encoding='utf-8'))
    persons = []
    for x in gl['nodes']:
        if x.get('kind') != 'professor':
            continue
        persons.append({'id': x['id'], 'kind': '교수', 'name': x['name'], 'affils': {uni_norm(u) for u in lit(x.get('affiliations')) if u},
                        'thesis_ids': lit(x.get('thesis_ids')), 'thesis_year': None, 'thesis_title': '', 'thesis_uni': ''})
    prof_thesis = {t for p in persons for t in p['thesis_ids']}
    for t in rp['theses']:
        if t['id'] in prof_thesis:
            continue   # 교수 본인 논문은 교수 쪽으로
        y = str(t.get('year', ''))[:4]
        persons.append({'id': t['id'], 'kind': '학위자', 'name': t.get('author', ''), 'affils': {uni_norm(t.get('university', ''))} if t.get('university') else set(),
                        'thesis_ids': [t['id']], 'thesis_year': int(y) if y.isdigit() else None, 'thesis_title': t.get('title', ''), 'thesis_uni': t.get('university', '')})
    th_meta = {t['id']: t for t in rp['theses']}
    for p in persons:
        if p['kind'] == '교수' and p['thesis_ids']:
            t = th_meta.get(p['thesis_ids'][0], {}); y = str(t.get('year', ''))[:4]
            p['thesis_year'] = int(y) if y.isdigit() else None; p['thesis_title'] = t.get('title', ''); p['thesis_uni'] = t.get('university', '')
    # ── 인물 → KAC (전거 A·B, 학위논문 A·A*)
    kac_of = collections.defaultdict(dict)
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/sparql/전거연결후보.csv', encoding='utf-8-sig', newline='')):
        if r['grade'] in ('A', 'B') and r['kac']:
            for pid in [x.strip() for x in r['our_ids'].split(';') if x.strip()]:
                kac_of[pid].setdefault(r['kac'], '전거' + r['grade'])
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/thesis/thesis_matches.csv', encoding='utf-8-sig', newline='')):
        if r['grade'] in ('A', 'A*') and r['kac']:
            for k in [x.strip() for x in r['kac'].split('|') if x.strip()]:
                kac_of[r['thesis_id']][k] = '학위논문' + r['grade']
    for p in persons:
        if p['kind'] == '교수':
            for t in p['thesis_ids']:
                for k, g in kac_of.get(t, {}).items():
                    kac_of[p['id']].setdefault(k, g)
    # ── KAC → 저작 제목(국중·ISNI), KRI
    kac_titles = collections.defaultdict(set); kac_kri = collections.defaultdict(set)
    wp = base / '참고문헌_구축/저자전거/works/nlk_works.csv'
    if wp.exists():
        for r in csv.DictReader(open(wp, encoding='utf-8-sig', newline='')):
            kac_titles[r['kac']].add(tnorm(r['title']))
            if r['rest']:
                kac_titles[r['kac']].add(tnorm(r['title'] + r['rest']))
    ip = base / '참고문헌_구축/저자전거/isni/isni_records.csv'
    if ip.exists():
        for r in csv.DictReader(open(ip, encoding='utf-8-sig', newline='')):
            if not r['kac']:
                continue
            for t in r['titles_head'].split(' | '):
                if t.strip():
                    kac_titles[r['kac']].add(tnorm(t))
            if r['kri']:
                kac_kri[r['kac']].add(r['kri'])
    kac_titles = {k: {t for t in v if len(t) >= 4} for k, v in kac_titles.items()}
    # ── 인물 × cretId 근거
    cand_rows = []; best = {}
    order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for p in persons:
        key = nm(p['name']); cids = sorted(by_name.get(key, ()))
        if not cids:
            best[p['id']] = {'grade': '—', 'cret_id': '', 'n_cands': 0}; continue
        kacs = kac_of.get(p['id'], {})
        my_titles = set().union(*(kac_titles.get(k, set()) for k in kacs)) if kacs else set()
        tg = grams(p['thesis_title']) if p['thesis_title'] else set()
        evid = []
        for cid in cids:
            c = clusters[cid]; insts = set(c['inst'])
            inst_hit = sorted(insts & p['affils'])
            titles_n = [tnorm(t) for t in c['titles']]
            title_hits = [c['titles'][i] for i, t in enumerate(titles_n) if t and t in my_titles]
            topic = max((jacc(tg, grams(t)) for t in c['titles'] if t), default=0.0) if tg else 0.0
            early = sum(1 for y in c['years'] if p['thesis_year'] and y < p['thesis_year'] - 10) if p['kind'] == '학위자' else 0
            evid.append({'cret_id': cid, 'inst_hit': inst_hit, 'title_hits': title_hits, 'topic': round(topic, 3), 'early': early, 'n_arts': len(c['arts']),
                         'years': ('%d–%d' % (min(c['years']), max(c['years']))) if c['years'] else '',
                         'insts': '; '.join('%s(%d)' % (k, v) for k, v in c['inst'].most_common(4)), 'names': '; '.join(c['names'])})
        n_inst = sum(1 for e in evid if e['inst_hit']); n_topic = sum(1 for e in evid if e['topic'] >= 0.25)
        for e in evid:
            if e['title_hits']:
                g = 'A'
            elif e['inst_hit'] and n_inst == 1:
                g = 'A' if (p['kind'] == '교수' or e['n_arts'] >= 2 or e['topic'] >= 0.15) else 'B'   # 학위자: 같은 대학 논문 1편·주제 무관이면 B
            elif e['inst_hit']:
                g = 'B'
            elif len(cids) == 1 and (p['kind'] == '교수' or e['topic'] >= 0.25):
                g = 'B'
            elif len(cids) == 1:
                g = 'C'
            elif e['topic'] >= 0.25 and n_topic == 1:
                g = 'B'
            else:
                g = 'D'
            if e['early'] >= 3 and g in ('B', 'C'):
                g = 'D'   # 학위 10년 전부터 활동 → 의심
            e['grade'] = g
        evid.sort(key=lambda e: (order[e['grade']], -len(e['title_hits']), -len(e['inst_hit']), -e['topic'], e['cret_id']))
        top = evid[0]; a_ids = [e['cret_id'] for e in evid if e['grade'] == 'A']
        multi = len(a_ids) > 1   # KCI 가 한 사람에게 cretId 를 여럿 주는 일이 흔함(소속·시기별) — 충돌이 아니라 동일인의 ID 묶음으로 본다
        mixed = multi and any(e['title_hits'] for e in evid if e['grade'] == 'A') and any(not e['title_hits'] for e in evid if e['grade'] == 'A')
        best[p['id']] = {'grade': top['grade'], 'cret_id': top['cret_id'], 'cret_ids_A': a_ids, 'n_cands': len(cids), 'multi': multi, 'mixed': mixed, 'top': top}
        kri = '; '.join(sorted(set().union(*(kac_kri.get(k, set()) for k in kacs)))) if kacs else ''
        for e in evid:
            cand_rows.append({'person_id': p['id'], 'kind': p['kind'], 'name': p['name'], 'our_affiliations': '; '.join(sorted(p['affils'])),
                              'thesis_year': p['thesis_year'] or '', 'thesis_title': p['thesis_title'],
                              'kac': '; '.join('%s(%s)' % (k, g) for k, g in sorted(kacs.items())), 'kri': kri,
                              'cret_id': e['cret_id'], 'grade': e['grade'], 'n_candidates_same_name': len(cids), 'multi_cret_A': int(multi), 'mixed_evidence': int(mixed),
                              'kci_names': e['names'], 'kci_institutions': e['insts'], 'kci_years': e['years'], 'kci_n_articles': e['n_arts'],
                              'inst_match': '; '.join(e['inst_hit']), 'title_matches': ' | '.join(e['title_hits'][:3]), 'n_title_matches': len(e['title_hits']),
                              'thesis_topic_overlap': e['topic'], 'articles_10y_before_phd': e['early'], 'decision': ''})
    cols = list(cand_rows[0].keys())
    with open(out / 'cret_candidates.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(cand_rows)
    best_rows = []
    for p in persons:
        b = best[p['id']]; t = b.get('top', {})
        best_rows.append({'person_id': p['id'], 'kind': p['kind'], 'name': p['name'], 'grade': b['grade'], 'cret_id': b['cret_id'],
                          'cret_ids_A': '; '.join(b.get('cret_ids_A', [])), 'n_candidates_same_name': b['n_cands'], 'multi_cret_A': int(b.get('multi', False)), 'mixed_evidence': int(b.get('mixed', False)),
                          'kci_institutions': t.get('insts', ''), 'kci_years': t.get('years', ''), 'kci_n_articles': t.get('n_arts', ''),
                          'inst_match': '; '.join(t.get('inst_hit', [])), 'n_title_matches': len(t.get('title_hits', [])), 'thesis_topic_overlap': t.get('topic', '')})
    with open(out / 'cret_best.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys())); w.writeheader(); w.writerows(best_rows)
    summ = {'generated': time.strftime('%Y-%m-%d %H:%M'),
            'collection': {'completed_articles': prog.get('completed_articles'), 'total_articles': prog.get('total_articles'), 'state': prog.get('state')},
            'kci_rows': len(rows), 'skipped_rows': dict(skipped), 'cret_clusters': len(clusters),
            'persons': {k: sum(1 for p in persons if p['kind'] == k) for k in ('교수', '학위자')},
            'grades': {k: dict(collections.Counter(best[p['id']]['grade'] for p in persons if p['kind'] == k)) for k in ('교수', '학위자')},
            'persons_with_kac': sum(1 for p in persons if kac_of.get(p['id'])),
            'title_evidence_persons': sum(1 for b in best.values() if b.get('top') and b['top']['title_hits']),
            'multi_cret_A_persons': sum(1 for b in best.values() if b.get('multi')), 'mixed_evidence_persons': sum(1 for b in best.values() if b.get('mixed'))}
    json.dump(summ, open(out / '요약.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    try:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = '요약'
        ws.append(['항목', '값'])
        for k, v in summ.items():
            ws.append([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
        ws.append([]); ws.append(['읽는 법', 'A: 저작 제목 일치 또는 소속 일치(유일) · B: 소속 일치(경합)/후보 하나+주제 · C: 후보 하나만 · D: 후보 여럿 · —: 이름 없음. 수집 중이면 — 와 후보 수는 늘어난다. decision 란에 확인/거부/보류.'])

        def sheet(title, rs, filt):
            s = wb.create_sheet(title); s.append(cols)
            for r in rs:
                if filt(r):
                    s.append([r[c] for c in cols])
        sheet('교수_검토(B·C·D)', cand_rows, lambda r: r['kind'] == '교수' and r['grade'] != 'A')
        sheet('학위자_검토(B·C·D)', cand_rows, lambda r: r['kind'] == '학위자' and r['grade'] != 'A')
        sheet('다중cretId_근거혼합(검토)', cand_rows, lambda r: r['mixed_evidence'] == 1 and r['grade'] == 'A')
        sheet('A_확정후보_전체', cand_rows, lambda r: r['grade'] == 'A')
        s = wb.create_sheet('인물별_최선'); s.append(list(best_rows[0].keys()))
        for r in best_rows:
            s.append(list(r.values()))
        wb.save(out / '연구자_검토표_KCI저자ID.xlsx')
    except ImportError:
        print('openpyxl 없음 — xlsx 생략')
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
