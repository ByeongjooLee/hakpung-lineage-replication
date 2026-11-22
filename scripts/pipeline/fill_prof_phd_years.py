"""교수 본인 박사학위 연도 보강 (2026-10-29)
1) 학맥 간선(교수가 target)·research-public 연도  2) NLK 박사논문 레코드(nlk_phd_theses.csv) 가운데 dcterms:creator KAC 가 교수의 전거 A급 KAC 인 것(확정)
3) KAC 가 없으면 dc:creator 이름이 같고 학과가 국문학 계열인 박사논문이 하나뿐일 때(추정)
출력: prof_phd_years.csv (professor_id, name, phd_year, source, kdm, university, dept, advisor_note, n_candidates)
"""
import argparse, csv, json, re, html, unicodedata, collections
from pathlib import Path

def nm(s):
    s = html.unescape(s or ''); s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\(（\[][^)）\]]*[\)）\]]', '', s)
    return re.sub(r'[\s,.·]+', '', s)
LIT_DEPT = re.compile(r'국어국문|국문학|국어 ?국문|國語國文|國文|문예창작|문학|文學|한국학|국어교육|國語敎育|비교문학|比較文學|한국어문')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); repo, base = Path(a.repo), Path(a.base)
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    rp = json.load(open(repo / 'site/data/research-public.json', encoding='utf-8')); th_year = {t['id']: t.get('year') for t in rp['theses']}
    profs = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    years = {}
    for e in gl['edges']:
        if e.get('target') in profs and str(e.get('year', '')).isdigit():
            years.setdefault(e['target'], {'phd_year': int(e['year']), 'source': 'genealogy_edge', 'kdm': '', 'university': e.get('university', ''), 'dept': '', 'advisor_note': e.get('advisor_raw', ''), 'n_candidates': 1})
    for p, node in profs.items():
        if p in years: continue
        for tid in node.get('thesis_ids') or []:
            y = th_year.get(tid)
            if str(y).isdigit(): years[p] = {'phd_year': int(y), 'source': 'research_public', 'kdm': '', 'university': '', 'dept': '', 'advisor_note': '', 'n_candidates': 1}; break
    # 전거 A급 KAC
    prof_kacs = collections.defaultdict(set)
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/sparql/전거연결후보.csv', encoding='utf-8-sig', newline='')):
        if r['grade'] == 'A' and '교수' in r['our_roles']:
            for pid in [x.strip() for x in r['our_ids'].split(';')]:
                if pid in profs: prof_kacs[pid].add(r['kac'])
    nlk = list(csv.DictReader(open(base / '참고문헌_구축/저자전거/thesis/nlk_phd_theses.csv', encoding='utf-8-sig', newline='')))
    by_kac = collections.defaultdict(list); by_name = collections.defaultdict(list)
    for t in nlk:
        for k in [x.strip() for x in t['kac'].split('|') if x.strip()]: by_kac[k].append(t)
        for c in [x.strip() for x in t['creator'].split('|') if x.strip()]: by_name[nm(c)].append(t)
    def yr(t):
        v = (t.get('year') or t.get('issued') or '')[:4]; return int(v) if v.isdigit() else None
    stats = collections.Counter(s['source'] for s in years.values())
    for p, node in profs.items():
        if p in years: continue
        cands = []
        for k in prof_kacs.get(p, ()):
            cands += [t for t in by_kac.get(k, []) if yr(t)]
        cands = list({t['kdm']: t for t in cands}.values())
        if cands:
            t = min(cands, key=yr)  # 같은 사람 박사논문이 둘이면 이른 것(국문학) — 드묾
            years[p] = {'phd_year': yr(t), 'source': 'nlk_kac', 'kdm': t['kdm'], 'university': t['pub'], 'dept': t['dept'], 'advisor_note': t['advisor_hangul'] or t['advisor_note'], 'n_candidates': len(cands)}
            continue
        nc = [t for t in by_name.get(nm(node['name']), []) if yr(t) and LIT_DEPT.search(t.get('dept', '') + ' ' + t.get('title', ''))]
        nc = list({t['kdm']: t for t in nc}.values())
        if len(nc) == 1:
            t = nc[0]; years[p] = {'phd_year': yr(t), 'source': 'nlk_name_unique_litdept', 'kdm': t['kdm'], 'university': t['pub'], 'dept': t['dept'], 'advisor_note': t['advisor_hangul'] or t['advisor_note'], 'n_candidates': 1}
        elif len(nc) > 1:
            years[p] = {'phd_year': '', 'source': 'ambiguous_%d' % len(nc), 'kdm': ' | '.join(t['kdm'] for t in nc[:5]), 'university': ' | '.join(t['pub'] for t in nc[:5]), 'dept': '', 'advisor_note': '', 'n_candidates': len(nc)}
    out = Path(a.out)
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['professor_id', 'name', 'affiliations', 'phd_year', 'source', 'kdm', 'university', 'dept', 'advisor_note', 'n_candidates'])
        for p, node in profs.items():
            y = years.get(p, {}); w.writerow([p, node['name'], ';'.join(node.get('affiliations') or []), y.get('phd_year', ''), y.get('source', 'none'), y.get('kdm', ''), y.get('university', ''), y.get('dept', ''), y.get('advisor_note', ''), y.get('n_candidates', 0)])
    final = collections.Counter(y['source'].split('_')[0] if y.get('phd_year') != '' and y.get('phd_year') is not None else 'none/ambiguous' for y in years.values())
    print('professors', len(profs), 'with year', sum(1 for y in years.values() if y.get('phd_year') not in ('', None)), dict(collections.Counter(y['source'] for y in years.values())))
    adv_new = sum(1 for p, y in years.items() if y['source'].startswith('nlk') and y.get('advisor_note'))
    print('nlk-filled professors with advisor note (candidate 상위 간선):', adv_new)

if __name__ == '__main__': main()
