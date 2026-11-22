"""국립중앙도서관 LOD SPARQL(https://lod.nl.go.kr/sparql, 키 불필요)에서 이름별 인물 전거 레코드를 CONSTRUCT 로 받아
nlk_snapshot_scan.py 와 같은 jsonl 스키마로 저장한다(현행 데이터 — 2026-10-08 갱신판 기준).

- 입력: names.csv (name 열). 출력(--out): raw/<이름>.rdf, nlk_sparql_matches.jsonl, nlk_sparql_matches.csv, progress.json, errors.jsonl
- foaf:name 또는 skos:altLabel 이 이름과 정확히 같은 nlon:Author 전부(nlon:create 저작 링크는 제외해 크기를 줄인다).
- SELECT 는 100행에서 잘리지만 CONSTRUCT 는 잘리지 않는다(2026-10-29 확인). 504 가 나면 이름의 인물 URI 를 SELECT+OFFSET 으로 모아 50명씩 나눠 받는다.
- 재실행하면 raw/ 에 있는 이름은 건너뛴다. http:// 는 연결이 거부되고 https:// 만 열린다.
"""
import argparse, csv, json, os, re, time, html, datetime as dt
import urllib.request, urllib.parse, urllib.error
from pathlib import Path
import xml.etree.ElementTree as ET

EP = 'https://lod.nl.go.kr/sparql'
UA = '<SITE>-research/1.0 (academic genealogy study)'
PFX = ('PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> '
       'PREFIX nlon: <http://lod.nl.go.kr/ontology/> ')
NS = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'nlon': 'http://lod.nl.go.kr/ontology/', 'foaf': 'http://xmlns.com/foaf/0.1/',
      'skos': 'http://www.w3.org/2004/02/skos/core#', 'schema': 'http://schema.org/', 'dc': 'http://purl.org/dc/elements/1.1/',
      'owl': 'http://www.w3.org/2002/07/owl#', 'dcterms': 'http://purl.org/dc/terms/', 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'}
def T(p, l): return '{%s}%s' % (NS[p], l)
F = {'name': T('foaf', 'name'), 'pref': T('skos', 'prefLabel'), 'alt': T('skos', 'altLabel'), 'birth': T('nlon', 'birthYear'), 'death': T('nlon', 'deathYear'),
     'field': T('nlon', 'fieldOfActivity'), 'job': T('schema', 'jobTitle'), 'corp': T('nlon', 'corporateName'), 'src': T('dc', 'source'),
     'isni': T('nlon', 'isni'), 'same': T('owl', 'sameAs'), 'mod': T('dcterms', 'modified'), 'lang': T('nlon', 'associatedLanguage'), 'gender': T('schema', 'gender')}
COLS = ['matched_name', 'kac', 'kac_uri', 'name', 'pref_label', 'alt_labels', 'birth_year', 'death_year', 'fields', 'jobs', 'corporate', 'isni', 'wikidata', 'viaf',
        'other_sameas_n', 'sources_n', 'sources_head', 'language', 'modified']

def q_sparql(query, fmt, timeout):
    url = EP + '?' + urllib.parse.urlencode({'query': query, 'format': fmt})
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': fmt, 'User-Agent': UA}), timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')

def lit(s): return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

def construct_name(name, timeout):
    # 이름 조건을 먼저 두어야 한다 — ?s a nlon:Author ; ?p ?o 를 앞에 두면 전체 스캔으로 이름 하나에 150초 걸린다(2026-10-29 확인, 바른 순서는 0.1~3초)
    q = PFX + ('CONSTRUCT { ?s ?p ?o } WHERE { { ?s foaf:name %s } UNION { ?s skos:altLabel %s } ?s a nlon:Author ; ?p ?o . '
               'FILTER(?p != nlon:create) }' % (lit(name), lit(name)))
    return q_sparql(q, 'application/rdf+xml', timeout)

def construct_fallback(name, timeout, interval):
    uris = []; off = 0
    while True:
        q = PFX + 'SELECT DISTINCT ?s WHERE { { ?s foaf:name %s } UNION { ?s skos:altLabel %s } ?s a nlon:Author } LIMIT 100 OFFSET %d' % (lit(name), lit(name), off)
        d = json.loads(q_sparql(q, 'application/sparql-results+json', timeout))
        b = d['results']['bindings']; uris += [x['s']['value'] for x in b]
        if len(b) < 100: break
        off += 100; time.sleep(interval)
    parts = []
    for i in range(0, len(uris), 40):
        vals = ' '.join('<%s>' % u for u in uris[i:i + 40])
        q = PFX + 'CONSTRUCT { ?s ?p ?o } WHERE { VALUES ?s { %s } ?s ?p ?o . FILTER(?p != nlon:create) }' % vals
        parts.append(q_sparql(q, 'application/rdf+xml', timeout)); time.sleep(interval)
    return parts

def year(v):
    m = re.match(r'(\d{3,4})', v or ''); return m.group(1) if m else ''

def parse_rdf(xml_text, name):
    root = ET.fromstring(xml_text.encode('utf-8'))
    recs = []
    for el in root:
        about = el.get(T('rdf', 'about'), '')
        if not about: continue
        rec = {'matched_name': name, 'kac_uri': about, 'kac': about.rsplit('/', 1)[-1], 'name': '', 'pref_label': '', 'alt_labels': [], 'birth_year': '', 'death_year': '',
               'fields': [], 'jobs': [], 'corporate': [], 'sources': [], 'isni': '', 'sameas': [], 'language': '', 'modified': '', 'gender': ''}
        for c in el:
            tx = (c.text or '').strip() or c.get(T('rdf', 'resource'), '')
            if c.tag == F['name']: rec['name'] = tx
            elif c.tag == F['pref']: rec['pref_label'] = tx
            elif c.tag == F['alt']: rec['alt_labels'].append(tx)
            elif c.tag == F['birth']: rec['birth_year'] = year(tx)
            elif c.tag == F['death']: rec['death_year'] = year(tx)
            elif c.tag == F['field']: rec['fields'].append(tx)
            elif c.tag == F['job']: rec['jobs'].append(tx)
            elif c.tag == F['corp']: rec['corporate'].append(tx)
            elif c.tag == F['src']: rec['sources'].append(tx)
            elif c.tag == F['isni']: rec['isni'] = tx
            elif c.tag == F['same']: rec['sameas'].append(tx)
            elif c.tag == F['lang']: rec['language'] = tx
            elif c.tag == F['gender']: rec['gender'] = tx
            elif c.tag == F['mod']: rec['modified'] = tx.split('^^')[0]
        rec['wikidata'] = next((u.rsplit('/', 1)[-1] for u in rec['sameas'] if 'wikidata.org' in u), '')
        rec['viaf'] = next((u.rstrip('/').rsplit('/', 1)[-1] for u in rec['sameas'] if 'viaf.org' in u), '')
        if not rec['isni']:
            rec['isni'] = next((u.rsplit('/', 1)[-1] for u in rec['sameas'] if 'isni.org' in u), '')
        recs.append(rec)
    return recs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=float, default=0.5); ap.add_argument('--timeout', type=float, default=180); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); raw = out / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    lock = out / '.running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        raise SystemExit('Output locked.')
    try:
        names = [r['name'] for r in csv.DictReader(open(a.names, encoding='utf-8-sig', newline=''))]
        done = {p.stem for p in raw.glob('*.rdf')}
        csv_path = out / 'nlk_sparql_matches.csv'; new = not csv_path.exists()
        attempted = 0; errs = 0; state = 'running'; t0 = time.time()
        def progress():
            (out / 'progress.json').write_text(json.dumps({'state': state, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'total_names': len(names),
                'completed_names': len(done), 'attempted_this_run': attempted, 'elapsed_s': round(time.time() - t0), 'pid': os.getpid()}, ensure_ascii=False, indent=2), encoding='utf-8')
        progress()
        with open(csv_path, 'a', encoding='utf-8', newline='') as fc, open(out / 'nlk_sparql_matches.jsonl', 'a', encoding='utf-8') as fj, \
             open(out / 'errors.jsonl', 'a', encoding='utf-8') as fe:
            w = csv.DictWriter(fc, fieldnames=COLS)
            if new: w.writeheader()
            for name in names:
                if name in done: continue
                if (out / 'STOP').exists(): state = 'stopped'; break
                if a.limit is not None and attempted >= a.limit: state = 'batch_finished'; break
                attempted += 1
                xmls = None; err = None
                for attempt in range(2):
                    try:
                        xmls = [construct_name(name, a.timeout)]; break
                    except urllib.error.HTTPError as e:
                        err = f'HTTP {e.code}'
                        if e.code in (504, 500, 503):
                            try:
                                xmls = construct_fallback(name, a.timeout, a.interval); err = None; break
                            except Exception as e2:
                                err = 'fallback ' + type(e2).__name__
                    except Exception as e:
                        err = type(e).__name__
                    time.sleep(a.interval * 6)
                if xmls is None:
                    errs += 1; fe.write(json.dumps({'name': name, 'error': err, 'at': dt.datetime.now(dt.timezone.utc).isoformat()}, ensure_ascii=False) + '\n'); fe.flush()
                    print(name, 'error', err, flush=True)
                    if errs >= 10: state = 'stopped_on_error'; break
                    time.sleep(a.interval * 6); continue
                errs = 0
                recs = []
                for x in xmls:
                    try: recs += parse_rdf(x, name)
                    except ET.ParseError as e:
                        fe.write(json.dumps({'name': name, 'error': 'ParseError', 'at': dt.datetime.now(dt.timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
                (raw / f'{name}.rdf').write_text('\n<!-- part -->\n'.join(xmls), encoding='utf-8')
                for r in recs:
                    w.writerow({'matched_name': name, 'kac': r['kac'], 'kac_uri': r['kac_uri'], 'name': r['name'], 'pref_label': r['pref_label'], 'alt_labels': ' | '.join(r['alt_labels']),
                                'birth_year': r['birth_year'], 'death_year': r['death_year'], 'fields': ' | '.join(r['fields']), 'jobs': ' | '.join(r['jobs']),
                                'corporate': ' | '.join(r['corporate']), 'isni': r['isni'], 'wikidata': r['wikidata'], 'viaf': r['viaf'],
                                'other_sameas_n': len([u for u in r['sameas'] if not any(k in u for k in ('wikidata.org', 'viaf.org', 'isni.org'))]),
                                'sources_n': len(r['sources']), 'sources_head': ' | '.join(r['sources'][:5]), 'language': r['language'], 'modified': r['modified']})
                    fj.write(json.dumps(r, ensure_ascii=False) + '\n')
                fc.flush(); fj.flush(); done.add(name)
                if attempted % 20 == 0: progress()
                time.sleep(a.interval)
            else:
                state = 'pass_finished'
        progress(); print('done', state, len(done), '/', len(names), flush=True)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
