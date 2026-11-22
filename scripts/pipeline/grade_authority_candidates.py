"""학위자·교수 ↔ 국중 전거(KAC/ISNI) 후보 등급표를 만든다. 자동 병합은 하지 않는다 — 연구자 검토용.

입력: nlk_snapshot_matches.jsonl (nlk_snapshot_scan.py), 저장소 research-public.json·genealogy-library.json
출력(--out): 전거연결후보.csv / .xlsx (openpyxl 있으면), 전거연결요약.json

등급(후보 레코드마다):
  A  문학 분야 + (교수 소속 대학이 전거 소속기관에 있음 | 박사논문 제목이 전거 저작에 있음)
  B  문학 분야 + (생년이 학위 연도와 맞음(학위 때 25~60세) | 학위 대학이 전거 소속기관에 있음) + 그 이름의 문학 분야 후보가 하나뿐
  C  문학 분야이나 동명 문학 후보가 여럿이거나 생년 없음
  D  문학 분야 아님(참고로만)
  —  전거 레코드 없음(이름 행만)
"""
import argparse, csv, json, re, html, unicodedata, collections
from pathlib import Path

LIT = re.compile(r'국어 ?국문|한국 ?문학|한국 ?시\b|한국 ?시\[|한국 ?소설|문학 ?평론|문학\(예술\)|비평|국어 ?교육|아동 ?문학|한국 ?수필|한국 ?희곡|문예 ?창작|현대 ?시|현대 ?소설|현대 ?문학|시인\[|소설가\[|평론가|문학\[')
NONLIT_ONLY = re.compile(r'영어 ?영문|중어 ?중문|일어 ?일문|불어 ?불문|독어 ?독문|노어|서어|중국 ?문학|영미 ?문학|영 ?문학|일본 ?문학|독일 ?문학|프랑스 ?문학|러시아 ?문학|중국어|일본어|영어\[')

def nm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·]+', '', s)

def tnorm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    return re.sub(r'[\s\W_]+', '', s).casefold()

def uni_key(s):
    """'성균관대학교 국어국문학과 (교수)' / '성균관대' → '성균관대'"""
    s = unicodedata.normalize('NFKC', s or '')
    s = s.replace('여자대학교', '여대').replace('대학교', '대')
    m = re.search(r'([가-힣A-Za-z]+?(?:여대|대))(?![학])', s)
    return m.group(1) if m else ''

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--matches', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rp = json.load(open(Path(a.repo) / 'site/data/research-public.json', encoding='utf-8'))
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8'))

    persons = {}  # key -> {name, roles:[...]} ; role = thesis / professor
    for t in rp['theses']:
        if not t.get('author'): continue
        k = nm(t['author'])
        p = persons.setdefault(k, {'name': k, 'thesis': [], 'professor': []})
        p['thesis'].append({'id': t.get('id', ''), 'title': t.get('title', ''), 'year': t.get('year'), 'university': t.get('university', ''), 'advisor': t.get('advisor', '')})
    for n in gl['nodes']:
        if n.get('kind') != 'professor' or not n.get('name'): continue
        k = nm(n['name'])
        p = persons.setdefault(k, {'name': k, 'thesis': [], 'professor': []})
        p['professor'].append({'id': n.get('id', ''), 'affiliations': n.get('affiliations') or [], 'advisee_count': n.get('advisee_thesis_count', 0),
                               'identity_status': n.get('identity_status', '')})

    recs = collections.defaultdict(list)
    for line in open(a.matches, encoding='utf-8'):
        r = json.loads(line); recs[r['matched_name']].append(r)

    rows = []; summary = collections.Counter(); best = {}
    for k in sorted(persons):
        p = persons[k]
        cands = recs.get(k, [])
        our_unis = {uni_key(x) for pr in p['professor'] for x in pr['affiliations']} - {''}
        our_titles = [tnorm(t['title']) for t in p['thesis'] if t.get('title')]
        thesis_years = [int(t['year']) for t in p['thesis'] if str(t.get('year', '')).isdigit()]
        lit_cands = []
        graded = []
        for r in cands:
            text = ' '.join(r['fields'] + r['jobs'])
            lit = bool(LIT.search(text)) and not (NONLIT_ONLY.search(text) and not re.search(r'국어 ?국문|한국 ?문학|한국 ?시|한국 ?소설|문학 ?평론', text))
            corp_unis = {uni_key(c) for c in r['corporate']} - {''}
            affil_hit = sorted(our_unis & corp_unis)
            src_norm = tnorm(' '.join(r['sources']))
            title_hit = [t['title'] for t in p['thesis'] if t.get('title') and len(tnorm(t['title'])) >= 6 and tnorm(t['title']) in src_norm]
            by = r.get('birth_year')
            birth_ok = None
            if by and by.isdigit() and thesis_years:
                ages = [y - int(by) for y in thesis_years]
                birth_ok = all(25 <= g <= 60 for g in ages)
            kri = any('KRI' in s or 'kri.go.kr' in s for s in r['sources'])
            thesis_unis = {uni_key(t['university']) for t in p['thesis']} - {''}
            thesis_uni_hit = sorted(thesis_unis & corp_unis)
            graded.append((r, lit, affil_hit, title_hit, birth_ok, kri, thesis_uni_hit))
            if lit: lit_cands.append(r['kac'])
        n_A = 0
        for r, lit, affil_hit, title_hit, birth_ok, kri, thesis_uni_hit in graded:
            if lit and (affil_hit or title_hit): g = 'A'; n_A += 1
            elif lit and (birth_ok or thesis_uni_hit) and len(lit_cands) == 1 and birth_ok is not False: g = 'B'
            elif lit: g = 'C'
            else: g = 'D'
            rows.append({
                'name': k, 'our_roles': ('학위자 ' if p['thesis'] else '') + ('교수' if p['professor'] else ''),
                'our_thesis': ' ; '.join(f"{t['year']} {t['university']} 《{t['title']}》" for t in p['thesis']),
                'our_affiliations': ' ; '.join(x for pr in p['professor'] for x in pr['affiliations']),
                'our_ids': ' ; '.join([t['id'] for t in p['thesis']] + [pr['id'] for pr in p['professor']]),
                'grade': g, 'kac': r['kac'], 'isni': r['isni'], 'wikidata': r['wikidata'], 'viaf': r['viaf'],
                'birth_year': r['birth_year'], 'death_year': r['death_year'], 'fields': ' | '.join(r['fields']), 'jobs': ' | '.join(r['jobs']),
                'corporate': ' | '.join(r['corporate']), 'affiliation_match': ' | '.join(affil_hit), 'thesis_title_in_sources': ' | '.join(title_hit),
                'thesis_university_match': ' | '.join(thesis_uni_hit),
                'birth_plausible': '' if birth_ok is None else ('예' if birth_ok else '아니오'), 'kri_source': '예' if kri else '',
                'n_lit_candidates_same_name': len(lit_cands), 'n_candidates_same_name': len(cands),
                'sources_head': ' | '.join(r['sources'][:6]), 'alt_labels': ' | '.join(r['alt_labels']), 'kac_uri': r['kac_uri'], 'decision': '',
            })
            summary[g] += 1
        if n_A > 1:
            for x in rows[-len(graded):]:
                if x['grade'] == 'A': x['conflict'] = f'A 후보 {n_A}건 — 동명이인 가능, 검토 필요'
        if not cands:
            rows.append({'name': k, 'our_roles': ('학위자 ' if p['thesis'] else '') + ('교수' if p['professor'] else ''),
                         'our_thesis': ' ; '.join(f"{t['year']} {t['university']} 《{t['title']}》" for t in p['thesis']),
                         'our_affiliations': ' ; '.join(x for pr in p['professor'] for x in pr['affiliations']),
                         'our_ids': ' ; '.join([t['id'] for t in p['thesis']] + [pr['id'] for pr in p['professor']]),
                         'grade': '—', 'n_candidates_same_name': 0, 'n_lit_candidates_same_name': 0, 'decision': ''})
            summary['—'] += 1
        gs = [x['grade'] for x in rows if x['name'] == k]
        best[k] = min(gs, key=lambda g: '—ABCD'.index(g) if g != '—' else 9)

    cols = ['name', 'our_roles', 'our_thesis', 'our_affiliations', 'our_ids', 'grade', 'kac', 'isni', 'wikidata', 'viaf', 'birth_year', 'death_year',
            'fields', 'jobs', 'corporate', 'affiliation_match', 'thesis_title_in_sources', 'thesis_university_match', 'birth_plausible', 'kri_source', 'conflict',
            'n_lit_candidates_same_name', 'n_candidates_same_name', 'sources_head', 'alt_labels', 'kac_uri', 'decision']
    rows.sort(key=lambda r: (r['name'], '—ABCD'.index(r['grade']) if r['grade'] != '—' else 9, r.get('kac', '')))
    with open(out / '전거연결후보.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in cols})
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = '전거연결후보'
        ws.append(cols)
        for c in ws[1]: c.font = Font(bold=True)
        fills = {'A': 'C6EFCE', 'B': 'FFEB9C', 'C': 'F2F2F2', 'D': 'FFFFFF', '—': 'FCE4D6'}
        for r in rows:
            ws.append([r.get(c, '') for c in cols])
            ws.cell(row=ws.max_row, column=cols.index('grade') + 1).fill = PatternFill('solid', fgColor=fills.get(r['grade'], 'FFFFFF'))
        ws.freeze_panes = 'G2'; ws.auto_filter.ref = ws.dimensions
        s = wb.create_sheet('요약')
        s.append(['등급', '후보 행 수']); [s.append([g, n]) for g, n in sorted(summary.items())]
        s.append([]); s.append(['이름별 최고 등급', '이름 수'])
        for g, n in sorted(collections.Counter(best.values()).items()): s.append([g, n])
        s.append([]); s.append(['등급 설명'])
        for line in __doc__.strip().splitlines()[5:]: s.append([line.strip()])
        wb.save(out / '전거연결후보.xlsx')
    except ImportError:
        print('openpyxl 없음 — csv 만 저장')
    best_counts = collections.Counter(best.values())
    summ = {'persons': len(persons), 'candidate_rows': dict(summary), 'best_grade_per_name': dict(best_counts),
            'names_with_A': sorted(k for k, g in best.items() if g == 'A')[:50]}
    (out / '전거연결요약.json').write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'persons': len(persons), 'rows': dict(summary), 'best_per_name': dict(best_counts)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
