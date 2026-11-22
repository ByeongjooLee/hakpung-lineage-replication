"""인물 식별표 — A 등급만 (2026-10-29, 연구자 지시 「확정되어 있는 A 만 먼저 자료 갱신」)

우리 인물(학맥 교수·학위자) ↔ 외부 식별자 가운데 **등급 A 로 확정된 것만** 한 표로 모은다.
  kac      국중 전거: 전거연결후보 A(이름·분야·소속/논문제목) 또는 학위논문 대응 A/A*(논문 레코드의 dcterms:creator)
  isni     그 KAC 전거 레코드의 nlon:isni (sparql/nlk_sparql_matches.csv)
  kri      ISNI SRU 레코드의 KRI 번호(있을 때만)
  cret_ids KCI 연구자 ID 집합 — match_cret_ids.py cret_best 의 grade A (저작 제목 일치 또는 소속 일치·유일)
B·C·D 는 넣지 않는다(검토표에서 연구자가 올린 뒤 다시 만든다). 이름만 같은 저자를 자동 병합하지 않는다는 원칙 그대로.
출력: 인물식별표_A.csv · 인물식별표_A.json · 요약 출력
"""
import argparse, ast, csv, json, collections, time
from pathlib import Path


def lit(v):
    try:
        return list(ast.literal_eval(v)) if isinstance(v, str) and v.startswith('[') else (v if isinstance(v, list) else [])
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); base, repo, out = Path(a.base), Path(a.repo), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    rp = json.load(open(repo / 'site/data/research-public.json', encoding='utf-8'))
    persons = {}
    for x in gl['nodes']:
        if x.get('kind') == 'professor':
            persons[x['id']] = {'person_id': x['id'], 'kind': '교수', 'name': x['name'], 'affiliations': ';'.join(lit(x.get('affiliations'))), 'thesis_ids': lit(x.get('thesis_ids'))}
    prof_thesis = {t for p in persons.values() for t in p['thesis_ids']}
    for t in rp['theses']:
        if t['id'] not in prof_thesis:
            persons[t['id']] = {'person_id': t['id'], 'kind': '학위자', 'name': t.get('author', ''), 'affiliations': t.get('university', ''), 'thesis_ids': [t['id']]}
    thesis_owner = {t: pid for pid, p in persons.items() for t in p['thesis_ids']}
    kac = collections.defaultdict(dict)   # pid -> {kac: 근거}
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/sparql/전거연결후보.csv', encoding='utf-8-sig', newline='')):
        if r['grade'] == 'A' and r['kac']:
            for pid in [x.strip() for x in r['our_ids'].split(';') if x.strip()]:
                owner = pid if pid in persons else thesis_owner.get(pid)
                if owner: kac[owner].setdefault(r['kac'], '전거A')
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/thesis/thesis_matches.csv', encoding='utf-8-sig', newline='')):
        if r['grade'] in ('A', 'A*') and r['kac']:
            owner = thesis_owner.get(r['thesis_id'])
            if owner:
                for k in [x.strip() for x in r['kac'].split('|') if x.strip()]: kac[owner].setdefault(k, '학위논문' + r['grade'])
    isni = collections.defaultdict(set)
    for r in csv.DictReader(open(base / '참고문헌_구축/저자전거/sparql/nlk_sparql_matches.csv', encoding='utf-8-sig', newline='')):
        if r['isni']: isni[r['kac']].add(r['isni'].replace(' ', ''))
    kri = collections.defaultdict(set)
    ip = base / '참고문헌_구축/저자전거/isni/isni_records.csv'
    if ip.exists():
        for r in csv.DictReader(open(ip, encoding='utf-8-sig', newline='')):
            if r['kac'] and r['kri']: kri[r['kac']].add(r['kri'])
    cret = {}
    cb = base / '참고문헌_구축/저자ID/식별/cret_best.csv'
    prog = json.load(open(base / '참고문헌_구축/저자ID/progress.json', encoding='utf-8'))
    for r in csv.DictReader(open(cb, encoding='utf-8-sig', newline='')):
        if r['grade'] == 'A': cret[r['person_id']] = [x for x in (r.get('cret_ids_A') or r['cret_id']).split('; ') if x]
    rows = []
    for pid, p in persons.items():
        ks = kac.get(pid, {}); cs = cret.get(pid, [])
        if not ks and not cs: continue
        rows.append({'person_id': pid, 'kind': p['kind'], 'name': p['name'], 'affiliations': p['affiliations'],
                     'kac': '; '.join(sorted(ks)), 'kac_evidence': '; '.join(ks[k] for k in sorted(ks)),
                     'isni': '; '.join(sorted(set().union(*(isni.get(k, set()) for k in ks)))) if ks else '',
                     'kri': '; '.join(sorted(set().union(*(kri.get(k, set()) for k in ks)))) if ks else '',
                     'cret_ids': '; '.join(cs), 'n_cret_ids': len(cs)})
    # 같은 KAC 를 두 인물이 나눠 가지면(동명 교수 쌍 김용성·김윤정·이화진 등) 둘 다 검토 대상 — 자동으로 한쪽에 주지 않는다
    owners = collections.defaultdict(set)
    for r in rows:
        for k in r['kac'].split('; '):
            if k: owners[k].add(r['person_id'])
    name_of = {r['person_id']: f"{r['name']}({r['affiliations']})" for r in rows}
    for r in rows:
        others = sorted({o for k in r['kac'].split('; ') if k for o in owners[k]} - {r['person_id']})
        r['kac_shared_with'] = '; '.join(name_of[o] for o in others)
        r['review'] = '동명 인물과 KAC 공유' if others else ('KAC 여럿(국중 중복 전거 가능)' if ';' in r['kac'] else '')
    rows.sort(key=lambda r: (r['kind'], r['name'], r['person_id']))
    with open(out / '인물식별표_A.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'cret_collection': f"{prog.get('completed_articles')}/{prog.get('total_articles')} ({prog.get('state')})", 'rows': len(rows)}
    for k in ('교수', '학위자'):
        rs = [r for r in rows if r['kind'] == k]
        summ[k] = {'persons': sum(1 for p in persons.values() if p['kind'] == k), 'with_kac': sum(1 for r in rs if r['kac']), 'with_isni': sum(1 for r in rs if r['isni']),
                   'with_kri': sum(1 for r in rs if r['kri']), 'with_cret': sum(1 for r in rs if r['cret_ids']), 'kac_and_cret': sum(1 for r in rs if r['kac'] and r['cret_ids']),
                   'multi_kac': sum(1 for r in rs if ';' in r['kac']), 'kac_shared_with_namesake': sum(1 for r in rs if r['kac_shared_with'])}
    json.dump({'summary': summ, 'persons': rows}, open(out / '인물식별표_A.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
