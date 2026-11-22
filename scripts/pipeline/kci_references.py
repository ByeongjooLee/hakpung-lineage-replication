"""KCI articleDetail reference collector. Small pilot by default; key never persisted."""
import argparse
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ENDPOINT = 'https://open.kci.go.kr/po/openapi/openApiSearch.kci'
LINK = 'https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId='

def local_retry(operation):
    """Bound retries for transient Windows sharing/permission conflicts only."""
    for attempt in range(6):
        try:
            return operation()
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))

def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (502, 503, 504) or attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))

def parse(text, source_id):
    root = ET.fromstring(text)
    records = root.findall('./outputData/record')
    if len(records) != 1:
        raise ValueError('missing_or_ambiguous_record')
    record = records[0]
    article = record.find('articleInfo')
    if article is None or article.get('article-id') != source_id:
        raise ValueError('source_id_mismatch')
    refs = record.find('referenceInfo')
    result = []
    for ordinal, ref in enumerate(refs.findall('reference') if refs is not None else [], 1):
        target = ref.get('arti-id', '')
        result.append({
            'source_kci_id': source_id, 'source_kci_url': LINK + source_id,
            'reference_order': ordinal, 'reference_id': ref.get('refebibl-id', ''),
            'target_kci_id': target, 'target_kci_url': LINK + target if target else '',
            'type_code': ref.get('type-code', ''), 'type_name': ref.get('type-name', ''),
            'title': ref.findtext('title', ''), 'author': ref.findtext('author', ''),
            'journal': ref.findtext('journal-name', ''),
            'publisher': ref.findtext('pubilisher', ref.findtext('publisher', '')),
            'year': ref.findtext('pubi-year', ''), 'volume': ref.findtext('volume', ''),
            'issue': ref.findtext('isseue', ref.findtext('issue', '')),
            'pages': ref.findtext('page', ''), 'doi': ref.findtext('doi', ''),
            'link_status': 'api_linked' if target else 'unlinked',
            'raw_reference_xml': ET.tostring(ref, encoding='unicode'),
        })
    return result, refs is not None

def export_by_title(out, rows):
    """Group by source article, sort by title, retain API reference order."""
    available = {p.stem for p in out.glob('ART*.json')}
    ordered = sorted((r for r in rows if r['kci_id'] in available), key=lambda r: (unicodedata.normalize('NFKC', r['title']).casefold(), r['kci_id']))
    index_fields = ['source_title','source_authors','source_year','source_journal','source_kci_id','source_kci_url','status','reference_count']
    with (out/'논문별_목록.csv.tmp').open('w',encoding='utf-8-sig',newline='') as ix, (out/'논문제목순_참고문헌.csv.tmp').open('w',encoding='utf-8-sig',newline='') as rf:
        iw=csv.DictWriter(ix,fieldnames=index_fields); iw.writeheader()
        rw=None
        for row in ordered:
            p=out/(row['kci_id']+'.json')
            if not p.exists():
                continue
            data=json.loads(p.read_text(encoding='utf-8'))
            source={'source_title':row['title'],'source_authors':row['authors'],'source_year':row['year'],
                    'source_journal':row['journal'],'source_kci_id':row['kci_id'],'source_kci_url':row['kci_url']}
            iw.writerow({**source,'status':data['status'],'reference_count':data['reference_count']})
            for ref in data['references']:
                item={**source,**{k:v for k,v in ref.items() if k not in source and k!='raw_reference_xml'}}
                item['reference_title']=item.pop('title')
                if rw is None:
                    rw=csv.DictWriter(rf,fieldnames=list(item)); rw.writeheader()
                rw.writerow(item)
    for name in ['논문별_목록.csv','논문제목순_참고문헌.csv']:
        local_retry(lambda: (out/(name+'.tmp')).replace(out/name))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='Existing CSV with kci_id and kci_url')
    ap.add_argument('--output', required=True, help='Separate reference collection directory')
    ap.add_argument('--limit', type=int, default=3, help='Maximum new requests this run (default 3)')
    ap.add_argument('--interval', type=float, default=2, help='Local precaution, NOT an official quota')
    args = ap.parse_args()
    if args.limit < 1 or args.interval < 1:
        ap.error('limit >= 1 and interval >= 1 required')
    key = os.environ.get('KCI_API_KEY') or getpass.getpass('KCI API key: ')
    if not key:
        ap.error('key required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('Output locked. Check running process before removing stale lock.')
    os.close(fd)
    try:
        with open(args.input, encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        rows = list({r['kci_id']: r for r in rows}.values())
        total = len(rows)
        existing = list(out.glob('ART*.json'))
        completed = len(existing)
        reference_total = sum(json.loads(p.read_text(encoding='utf-8'))['reference_count'] for p in existing)
        state = 'running'
        def progress():
            value = {'state': state, 'updated_at': datetime.now(timezone.utc).isoformat(),
                     'total_articles': total, 'completed_articles': completed,
                     'reference_rows': reference_total, 'attempted_this_run': attempted,
                     'input': str(Path(args.input).resolve()), 'pid': os.getpid()}
            temp = out / 'progress.tmp'
            local_retry(lambda: temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'))
            local_retry(lambda: temp.replace(out / 'progress.json'))
        done = attempted = 0
        progress()
        for row in rows:
            aid = row['kci_id']
            if not re.fullmatch(r'ART\d+', aid):
                continue
            result_file = out / (aid + '.json')
            if result_file.exists():
                continue
            if attempted >= args.limit or (out / 'STOP').exists():
                break
            if attempted:
                time.sleep(args.interval)
            attempted += 1
            stamp = datetime.now(timezone.utc).isoformat()
            params = {'key': key, 'apiCode': 'articleDetail', 'id': aid}
            try:
                raw = fetch(ENDPOINT + '?' + urllib.parse.urlencode(params))
                text = raw.decode('utf-8-sig').replace(key, '[REDACTED]')
                # The API echoes the key in inputData. Persist only the redacted response.
                local_retry(lambda: (out / (aid + '.xml')).write_text(text, encoding='utf-8'))
                refs, has_container = parse(text, aid)
                result = {'source_kci_id': aid, 'source_kci_url': row.get('kci_url') or LINK + aid,
                          'source_title': row['title'], 'source_authors': row['authors'],
                          'source_year': row['year'], 'source_journal': row['journal'],
                          'retrieved_at': stamp, 'endpoint': ENDPOINT, 'apiCode': 'articleDetail',
                          'redacted_xml_sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'status': 'references_returned' if refs else 'empty_referenceInfo' if has_container else 'referenceInfo_absent',
                          'reference_count': len(refs), 'references': refs}
                temp = result_file.with_suffix('.tmp')
                local_retry(lambda: temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'))
                local_retry(lambda: temp.replace(result_file))
                done += 1
                completed += 1
                reference_total += len(refs)
                progress()
                if done % 100 == 0:
                    export_by_title(out, rows)
                print(aid, result['status'], len(refs), flush=True)
            except Exception as exc:
                # Do not log exception URLs: they may contain the key. Stop on first error.
                error = {'source_kci_id': aid, 'retrieved_at': stamp, 'error_type': type(exc).__name__}
                if isinstance(exc, urllib.error.HTTPError):
                    error['http_status'] = exc.code
                with (out / 'errors.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(error) + '\n')
                print(aid, 'stopped_on_error', error['error_type'], flush=True)
                state = 'stopped_on_error'
                break
        if state == 'running':
            state = 'pass_finished' if completed == total else 'stopped' if (out / 'STOP').exists() else 'batch_finished'
        progress()
        export_by_title(out, rows)
        with (out / 'references.csv.tmp').open('w', encoding='utf-8-sig', newline='') as f:
            writer = None
            for path in sorted(out.glob('ART*.json')):
                refs = json.loads(path.read_text(encoding='utf-8'))['references']
                if refs and writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(refs[0]))
                    writer.writeheader()
                if refs:
                    writer.writerows(refs)
        local_retry(lambda: (out / 'references.csv.tmp').replace(out / 'references.csv'))
        print(json.dumps({'attempted_this_run': attempted, 'saved_this_run': done,
                          'reference_rows_in_output': reference_total}))
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
