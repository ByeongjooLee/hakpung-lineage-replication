# -*- coding: utf-8 -*-
"""새 학맥(2026-11-22 판)에 맞춰 분석 표본을 다시 잡는다 (2026-11-22, 연구자 「검수해 추가한 학맥 확인하고 논문 분석 다시」).

교수 묶음(KCI 논문 벡터)은 그대로 두고 표본만 바꾼다:
  - 교수: 옛 표본 교수 가운데 새 학맥에 교수 노드로 남은 사람
  - 논문: 그 교수들이 지도한 새 학맥의 간선. 옛 표본에 있던 간선은 벡터를 그대로 쓰고, 새 간선은 같은 규칙(제목 + 공개 태그)으로 임베딩
  - 본문 기반 표현(tfidf·lsa·lda 등)은 새 간선의 벡터를 만들 수 없으므로 남은 옛 간선만

  py -X utf8 resample.py spec  --old <옛 bge 폴더> --repo <새 학맥 폴더> --hanja <hanja_norm.py 가 있는 폴더> --out <작업 폴더>
  <venv> -X utf8 resample.py embed --spec <작업 폴더>/sample_spec.json --kind bge|onnx --model <경로> --name <이름> [--prefix 'query: ']
  py -X utf8 resample.py build --old <옛 표현 폴더> --spec <…>/sample_spec.json --out <새 표현 폴더> [--newvecs <…>/newvecs_<이름>.npy] [--only-old]
"""
import argparse, json, re, sys, unicodedata, collections
from pathlib import Path
import numpy as np


def uni_key(s):
    s = unicodedata.normalize('NFKC', s or '').replace('여자대학교', '여대').replace('대학교', '대')
    m = re.search(r'([가-힣A-Za-z]+?(?:여대|대))(?![학])', s)
    return m.group(1) if m else ''


def cmd_spec(a):
    sys.path.insert(0, a.hanja)
    from hanja_norm import h2k
    old = Path(a.old); repo = Path(a.repo); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(old / 'vectors.npz', allow_pickle=True); old_pids = [str(x) for x in z['pids']]; old_eids = [str(x) for x in z['edge_ids']]
    gl = json.load(open(repo / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs = {n['id']: n for n in gl['nodes'] if n.get('kind') == 'professor'}
    tp = json.load(open(repo / 'site/data/thesis-topics-public.json', encoding='utf-8')); ttags = {t['id']: t for t in tp['topics']}
    keep = [p for p in old_pids if p in profs]; keepset = set(keep); old_idx = {e: i for i, e in enumerate(old_eids)}
    edges = [e for e in gl['edges'] if e.get('source') in keepset and e.get('title')]
    def text(e):
        tt = ttags.get(e.get('thesis_id'), {}); extra = ' '.join((tt.get('people') or []) + (tt.get('concepts') or []))
        return h2k(e['title']) + ((' — ' + extra) if extra else '')
    rows = []; seen = set()
    for e in edges:
        if e['id'] in seen: continue
        seen.add(e['id'])
        rows.append({'edge_id': e['id'], 'professor_id': e['source'], 'student': e['target'], 'thesis_id': e.get('thesis_id'), 'year': e.get('year', ''),
                     'university': uni_key(e.get('university', '')), 'identity': (e.get('student_identity') or {}).get('status', ''),
                     'old_row': old_idx.get(e['id'], -1), 'text': text(e), 'in_corpus': e.get('thesis_id') in ttags, 'basis': e.get('basis', '')})
    # 옛 순서 먼저, 새 간선은 뒤에
    rows.sort(key=lambda r: (r['old_row'] < 0, r['old_row'] if r['old_row'] >= 0 else 0))
    new_rows = [r for r in rows if r['old_row'] < 0]; old_rows = [r for r in rows if r['old_row'] >= 0]
    check = old_rows[:: max(1, len(old_rows) // 40)][:40]     # 임베딩 규칙이 옛것과 같은지 확인할 옛 논문
    spec = {'as_of': gl.get('as_of'), 'old': str(old), 'professors': {p: {'name': profs[p]['name'], 'universities': sorted({uni_key(x) for x in (profs[p].get('affiliations') or [])} - {''})} for p in keep},
            'pids': keep, 'dropped_pids': [p for p in old_pids if p not in keepset], 'edges': rows,
            'dropped_edges': [e for e in old_eids if e not in seen], 'check_rows': [r['old_row'] for r in check], 'check_texts': [r['text'] for r in check]}
    json.dump(spec, open(out / 'sample_spec.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('교수', len(old_pids), '→', len(keep), '| 간선', len(old_eids), '→', len(rows), '(옛', len(old_rows), '+ 새', len(new_rows), ') | 빠진 옛 간선', len(spec['dropped_edges']))
    print('새 간선 중 본문 말뭉치에 있는', sum(r['in_corpus'] for r in new_rows))
    print('새 간선 근거', collections.Counter(r['basis'] for r in new_rows).most_common(8))


def cmd_embed(a):
    spec = json.load(open(a.spec, encoding='utf-8')); out = Path(a.spec).parent
    texts = [r['text'] for r in spec['edges'] if r['old_row'] < 0] + spec['check_texts']
    if a.prefix: texts = [a.prefix + t for t in texts]
    import torch  # noqa
    if a.kind == 'bge':
        sys.path.insert(0, r'<HOME>\Downloads\drive-download-20260918T092441Z-1-002\작업도구\추천')
        from embed_local_bge import LocalBGE
        model = LocalBGE(max_len=a.max_len, threads=4); v = np.asarray(model.encode(texts, batch=64), dtype=np.float32)
    else:
        sys.path.insert(0, a.tools)
        from embed_variants import OnnxEmbedder
        model = OnnxEmbedder(a.model, max_len=512); v = np.asarray(model.encode(texts, batch=32), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    n_new = len(texts) - len(spec['check_texts'])
    np.save(out / f'newvecs_{a.name}.npy', v[:n_new]); np.save(out / f'checkvecs_{a.name}.npy', v[n_new:])
    print(a.name, 'new', n_new, 'check', len(spec['check_texts']), v.shape, flush=True)


def cmd_build(a):
    spec = json.load(open(a.spec, encoding='utf-8')); old = Path(a.old); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(old / 'vectors.npz', allow_pickle=True); old_pids = [str(x) for x in z['pids']]; old_eids = [str(x) for x in z['edge_ids']]
    oi = {e: i for i, e in enumerate(old_eids)}; pi = {p: i for i, p in enumerate(old_pids)}
    pids = [p for p in spec['pids'] if p in pi]; assert len(pids) == len(spec['pids']), '교수 목록이 옛 표현과 다르다'
    STU = z['stu']; newv = None
    if a.newvecs:
        newv = np.load(a.newvecs)
        if a.check:
            cv = np.load(a.check); rows = spec['check_rows']; bge_old = np.load(Path(spec['old']) / 'vectors.npz', allow_pickle=True)['edge_ids']
            cs = []
            for k, r in enumerate(rows):
                eid = str(bge_old[r])
                if eid in oi:
                    o = STU[oi[eid]].astype(np.float64); o /= np.linalg.norm(o) + 1e-9; cs.append(float(o @ cv[k]))
            print('임베딩 규칙 확인(옛 벡터와 다시 임베딩한 벡터의 코사인): 최소 %.4f 평균 %.4f (n=%d)' % (min(cs), sum(cs) / len(cs), len(cs)))
    rows = []; vecs = []; k_new = 0
    for r in spec['edges']:
        if r['old_row'] < 0:
            if newv is not None and not a.only_old: rows.append(r); vecs.append(newv[k_new].astype(STU.dtype))
            k_new += 1
        elif r['edge_id'] in oi:
            rows.append(r); vecs.append(STU[oi[r['edge_id']]])
    stu = np.vstack(vecs); prof = z['prof'][[pi[p] for p in pids]]
    save = dict(stu=stu, prof=prof, pids=np.array(pids), edge_ids=np.array([r['edge_id'] for r in rows]), prof_ndocs=z['prof_ndocs'][[pi[p] for p in pids]])
    if 'docs_vec' in z.files:
        remap = {pi[p]: j for j, p in enumerate(pids)}; dp = z['docs_pid']; m = np.array([int(x) in remap for x in dp])
        save.update(docs_vec=z['docs_vec'][m], docs_pid=np.array([remap[int(x)] for x in dp[m]], dtype=np.int64), docs_year=z['docs_year'][m])
    np.savez_compressed(out / 'vectors.npz', **save)
    ometa = json.load(open(old / 'vectors_meta.json', encoding='utf-8'))
    meta = {'edges': [{k: r[k] for k in ('edge_id', 'professor_id', 'student', 'thesis_id', 'year', 'university', 'identity')} for r in rows],
            'professors': {p: {'name': spec['professors'][p]['name'], 'universities': spec['professors'][p]['universities'], 'n_docs': ometa['professors'][p]['n_docs']} for p in pids},
            'genealogy_as_of': spec['as_of'], 'resampled_from': str(old)}
    json.dump(meta, open(out / 'vectors_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(out.name, '교수', len(pids), '논문', len(rows), '(새 간선', sum(r['old_row'] < 0 for r in rows), ')', 'stu', stu.shape, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('spec'); s.add_argument('--old', required=True); s.add_argument('--repo', required=True); s.add_argument('--hanja', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('embed'); s.add_argument('--spec', required=True); s.add_argument('--kind', required=True); s.add_argument('--model'); s.add_argument('--name', required=True); s.add_argument('--prefix', default=''); s.add_argument('--tools', default=''); s.add_argument('--max-len', type=int, default=512)
    s = sub.add_parser('build'); s.add_argument('--old', required=True); s.add_argument('--spec', required=True); s.add_argument('--out', required=True); s.add_argument('--newvecs'); s.add_argument('--check'); s.add_argument('--only-old', action='store_true')
    a = ap.parse_args(); {'spec': cmd_spec, 'embed': cmd_embed, 'build': cmd_build}[a.cmd](a)
