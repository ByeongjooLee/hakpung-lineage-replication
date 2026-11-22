"""KCI 논문 웹페이지에서 저자별 KCI 연구자 ID(cretId)·표기·소속만 수집한다.

- 입력: targets.csv (kci_id 열). 기본은 참고문헌_구축/저자ID/targets.csv
- 출력(--out 폴더):
    author_blocks.jsonl  논문별 저자·소속 블록 HTML 조각(재파싱용. 초록·본문 없음)
    author_ids.csv       파싱 결과: kci_id, author_order, cret_id, author_text, name_ko, name_en, sup_num, institution, retrieved_at
    errors.jsonl, progress.json, .running.lock, STOP(있으면 협조적 중단)
- 로그인·API 키·자격증명을 쓰지 않는다. 공개 논문 페이지만 요청한다(기본 2초 간격).
- 다시 실행하면 author_blocks.jsonl 에 있는 논문은 건너뛴다(재개).
- 참고문헌 수집기(kci_references.py)와 폴더·잠금이 다르다. 같은 폴더에 두 개를 돌리지 말 것.
"""
import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ARTICLE_URL = 'https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId={aid}'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) <SITE>-research/1.0 (academic genealogy study; 2s interval)'
CSV_FIELDS = ['kci_id', 'author_order', 'cret_id', 'author_text', 'name_ko', 'name_en', 'sup_num', 'institution', 'retrieved_at']


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_text(s):
    s = re.sub(r'<[^>]+>', ' ', s or '')
    s = html.unescape(html.unescape(s))  # KCI 는 이름 엔티티를 이중으로 인코딩하기도 한다
    s = s.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def extract_blocks(page):
    """<article class="report-detail ...> 안의 저자·소속 div 조각을 돌려준다. 없으면 None."""
    i = page.find('<article class="report-detail')
    if i < 0:
        return None
    seg = page[i:]
    j = seg.find('</article>')
    seg = seg[:j] if j > 0 else seg[:20000]
    m_auth = re.search(r'<div class="author">(.*?)</div>\s*(?=<!--|<div class="attach")', seg, re.S)
    m_att = re.search(r'<div class="attach">(.*?)</div>', seg, re.S)
    return {
        'author_html': re.sub(r'\s+', ' ', m_auth.group(1)).strip() if m_auth else '',
        'attach_html': re.sub(r'\s+', ' ', m_att.group(1)).strip() if m_att else '',
    }


def parse_authors(author_html, attach_html):
    affil = {}
    for num, inst in re.findall(r'<p>\s*<sup class="num">(\d+)</sup>(.*?)</p>', attach_html or '', re.S):
        affil[num] = clean_text(inst)
    entries = []
    cur = None
    tokens = re.finditer(r'<a\b[^>]*>.*?</a>|<sup class="num">\d+</sup>|<[^>]+>|[^<]+', author_html or '', re.S)
    for t in tokens:
        tok = t.group(0)
        if tok.startswith('<a'):
            m = re.search(r'cretId=(CRT\d+)', tok)
            inner = re.search(r'>(.*)</a>', tok, re.S).group(1)
            sup = re.search(r'<sup class="num">(\d+)</sup>', inner)
            cur = {'cret_id': m.group(1) if m else '', 'author_text': clean_text(inner), 'sup_num': sup.group(1) if sup else ''}
            entries.append(cur)
        elif tok.startswith('<sup'):
            num = re.search(r'(\d+)', tok).group(1)
            if cur is not None and not cur['sup_num']:
                cur['sup_num'] = num
        elif tok.startswith('<'):
            continue
        else:
            text = clean_text(tok)
            for piece in re.split(r'[,;]', text):
                piece = piece.strip(' ,;')
                if piece:
                    cur = {'cret_id': '', 'author_text': piece, 'sup_num': ''}
                    entries.append(cur)
    for k, e in enumerate(entries, 1):
        e['author_order'] = k
        txt = e['author_text']
        if '/' in txt:
            ko, en = txt.split('/', 1)
            e['name_ko'], e['name_en'] = ko.strip(), en.strip()
        else:
            e['name_ko'], e['name_en'] = txt, ''
        if e['sup_num'] in affil:
            e['institution'] = affil[e['sup_num']]
        elif len(entries) == 1 and len(affil) == 1:
            e['institution'] = next(iter(affil.values()))
        else:
            e['institution'] = ''
    return entries


def fetch(aid, timeout):
    req = urllib.request.Request(ARTICLE_URL.format(aid=aid), headers={'User-Agent': UA, 'Accept-Language': 'ko'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8', 'replace')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', required=True, help='kci_id 열이 있는 CSV')
    ap.add_argument('--out', required=True, help='출력 폴더')
    ap.add_argument('--interval', type=float, default=2.0, help='요청 사이 대기(초). 자체 예방 조치')
    ap.add_argument('--limit', type=int, default=None, help='이번 실행의 최대 요청 수(기본 제한 없음)')
    ap.add_argument('--timeout', type=float, default=30.0)
    ap.add_argument('--max-consecutive-errors', type=int, default=10, help='연속 오류가 이 수에 이르면 멈춘다')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{os.getpid()} {now_iso()}\n'.encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit('Output locked. Check running process before removing stale lock.')

    try:
        with open(args.targets, encoding='utf-8-sig', newline='') as f:
            targets = [r['kci_id'].strip() for r in csv.DictReader(f) if r.get('kci_id', '').strip()]
        blocks_path = out / 'author_blocks.jsonl'
        done = set()
        if blocks_path.exists():
            with open(blocks_path, encoding='utf-8') as f:
                for line in f:
                    try:
                        done.add(json.loads(line)['kci_id'])
                    except Exception:
                        pass
        csv_path = out / 'author_ids.csv'
        new_csv = not csv_path.exists()
        total = len(targets)
        attempted = 0
        errors_in_row = 0
        state = 'running'

        def progress(extra=None):
            d = {'state': state, 'updated_at': now_iso(), 'total_articles': total, 'completed_articles': len(done),
                 'attempted_this_run': attempted, 'pid': os.getpid(), 'targets': str(args.targets)}
            if extra:
                d.update(extra)
            tmp = out / 'progress.tmp'
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(out / 'progress.json')

        progress()
        with open(blocks_path, 'a', encoding='utf-8') as fb, open(csv_path, 'a', encoding='utf-8', newline='') as fc, \
                open(out / 'errors.jsonl', 'a', encoding='utf-8') as fe:
            w = csv.DictWriter(fc, fieldnames=CSV_FIELDS)
            if new_csv:
                w.writeheader()
            for aid in targets:
                if aid in done:
                    continue
                if (out / 'STOP').exists():
                    state = 'stopped'
                    break
                if args.limit is not None and attempted >= args.limit:
                    state = 'batch_finished'
                    break
                attempted += 1
                status = None
                page = None
                err = None
                for attempt in range(3):
                    try:
                        status, page = fetch(aid, args.timeout)
                        break
                    except urllib.error.HTTPError as e:
                        err = {'error_type': 'HTTPError', 'http_status': e.code}
                        if e.code in (404, 400):
                            break
                    except Exception as e:  # timeout, URLError 등
                        err = {'error_type': type(e).__name__}
                    time.sleep(args.interval * (attempt + 2))
                blocks = extract_blocks(page) if page is not None else None
                if page is not None and blocks is None:
                    err = {'error_type': 'NoArticleBlock', 'http_status': status, 'length': len(page)}
                if blocks is None:
                    err.update({'kci_id': aid, 'retrieved_at': now_iso()})
                    fe.write(json.dumps(err, ensure_ascii=False) + '\n'); fe.flush()
                    errors_in_row += 1
                    print(aid, 'error', err['error_type'], flush=True)
                    if errors_in_row >= args.max_consecutive_errors:
                        state = 'stopped_on_error'
                        break
                else:
                    errors_in_row = 0
                    ts = now_iso()
                    rec = {'kci_id': aid, 'retrieved_at': ts, 'http_status': status, **blocks}
                    fb.write(json.dumps(rec, ensure_ascii=False) + '\n'); fb.flush()
                    for e in parse_authors(blocks['author_html'], blocks['attach_html']):
                        w.writerow({'kci_id': aid, 'retrieved_at': ts, **{k: e.get(k, '') for k in CSV_FIELDS if k not in ('kci_id', 'retrieved_at')}})
                    fc.flush()
                    done.add(aid)
                if attempted % 50 == 0:
                    progress()
                time.sleep(args.interval)
            else:
                state = 'pass_finished' if len(done) >= total else 'batch_finished'
        progress()
        print('done', state, len(done), '/', total, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
