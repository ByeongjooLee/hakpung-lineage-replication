"""계보 뿌리별 토픽 표와 η² 를 학맥(저장소 genealogy-library.json)만 바꿔 다시 계산한다 (2026-11-14).

tfidf_topic_replication.py 의 topic_tables() 가운데 학맥에 기대는 부분(lineage_topics_k50.csv, eta2_k50.json)을 같은 식·같은 난수 순서로 옮긴 것.
θ 는 이미 계산된 LDA 벡터(학풍_TFIDF_토픽/lda50/vectors.npz, stu = sqrt θ)에서 읽는다. 토큰·본문은 쓰지 않는다.
실행: py -X utf8 topic_lineage_recompute.py --lda <…/학풍_TFIDF_토픽/lda50> --repo <저장소 또는 site/data 를 가진 폴더> --out <폴더> [--seed 42]
"""
import argparse, csv, json, collections
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lda', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); K = 50
    z = np.load(Path(a.lda) / 'vectors.npz', allow_pickle=True); meta = json.load(open(Path(a.lda) / 'vectors_meta.json', encoding='utf-8'))
    edges = meta['edges']; assert [e['edge_id'] for e in edges] == list(z['edge_ids'])
    TH_s = z['stu'].astype(np.float64) ** 2
    gl = json.load(open(Path(a.repo) / 'site/data/genealogy-library.json', encoding='utf-8'))
    profs_all = {x['id']: x for x in gl['nodes'] if x.get('kind') == 'professor'}
    advisor_of = {}
    for e in gl['edges']:
        if e.get('target') in profs_all and e.get('source') in profs_all and e['target'] != e['source']: advisor_of.setdefault(e['target'], e['source'])
    def root(p):
        seen = set(); cur = p
        while cur in advisor_of and cur not in seen: seen.add(cur); cur = advisor_of[cur]
        return cur
    pinfo = meta['professors']
    def name_of(p): return (profs_all.get(p) or {}).get('name') or (pinfo.get(p) or {}).get('name') or p
    roots = [root(e['professor_id']) for e in edges]; unis = [e['university'] for e in edges]

    def group_table(labels, fname, min_n=10):
        cnt = collections.Counter(labels); overall = TH_s.mean(axis=0)
        with open(out / fname, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(['group', 'n_theses'] + [f't{k}' for k in range(K)] + ['top_distinct_topics'])
            for g, n in sorted(cnt.items(), key=lambda x: (-x[1], str(x[0]))):
                if n < min_n: continue
                m = TH_s[[i for i, l in enumerate(labels) if l == g]].mean(axis=0); ratio = (m + 1e-4) / (overall + 1e-4)
                cand = [k for k in np.argsort(-ratio) if overall[k] >= 0.01]   # 전체 비중 1% 이상 토픽에서만(잔 토픽은 비율이 요동친다 — 원 보고서와 같은 규칙)
                top = [f't{k}×{ratio[k]:.1f}' for k in cand[:4]]
                w.writerow([g, n] + [round(float(x), 4) for x in m] + [' '.join(top)])
    group_table([name_of(r) for r in roots], f'lineage_topics_k{K}.csv'); group_table(unis, f'university_topics_k{K}.csv')

    V = np.sqrt(np.clip(TH_s, 0, None)); V = V / np.linalg.norm(V, axis=1, keepdims=True); rng = np.random.default_rng(a.seed)
    def eta2(labels):
        lab = np.array(labels); cnt = collections.Counter(labels); keep = np.array([cnt[l] >= 5 for l in labels])
        Vk, lk = V[keep], lab[keep]; mu = Vk.mean(axis=0); sst = float(((Vk - mu) ** 2).sum())
        def ssb(l):
            s = 0.
            for g in set(l):
                m = Vk[l == g]; s += len(m) * float(((m.mean(axis=0) - mu) ** 2).sum())
            return s
        obs = ssb(lk) / sst; null = [ssb(rng.permutation(lk)) / sst for _ in range(300)]
        return {'eta2': obs, 'null_mean': float(np.mean(null)), 'null_sd': float(np.std(null)), 'p': float(np.mean([x >= obs for x in null])), 'n_theses': int(keep.sum()), 'n_groups': len(set(lk))}
    advisors = [e['professor_id'] for e in edges]; root_l = [name_of(r) for r in roots]; combo = [f'{u}|{r}' for u, r in zip(unis, root_l)]
    def eta2_inc(coarse, fine):
        coarse, fine = np.array(coarse), np.array(fine); comb = np.array([f'{c}|{f}' for c, f in zip(coarse, fine)])
        cnt = collections.Counter(comb.tolist()); keep = np.array([cnt[x] >= 5 for x in comb]); Vk = V[keep]; mu = Vk.mean(axis=0); sst = float(((Vk - mu) ** 2).sum())
        def e2(l):
            s = 0.
            for g in set(l.tolist()):
                m = Vk[l == g]; s += len(m) * float(((m.mean(axis=0) - mu) ** 2).sum())
            return s / sst
        ck, fk = coarse[keep], fine[keep]; obs = e2(comb[keep]) - e2(ck); null = []
        for _ in range(300):
            f2 = fk.copy()
            for g in set(ck.tolist()):
                ii = np.flatnonzero(ck == g); f2[ii] = f2[rng.permutation(ii)]
            null.append(e2(np.array([f'{c}|{f}' for c, f in zip(ck, f2)])) - e2(ck))
        return {'increment': obs, 'null_mean': float(np.mean(null)), 'null_sd': float(np.std(null)), 'p': float(np.mean([x >= obs for x in null])), 'n_theses': int(keep.sum())}
    res = {'advisor': eta2(advisors), 'lineage_root': eta2(root_l), 'university': eta2(unis), 'university_x_lineage': eta2(combo),
           'university_beyond_lineage': eta2_inc(root_l, unis), 'lineage_beyond_university': eta2_inc(unis, root_l), 'advisor_beyond_university': eta2_inc(unis, advisors)}
    json.dump(res, open(out / f'eta2_k{K}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()} for k, v in res.items()}, ensure_ascii=False))
    print('roots', len(set(roots)), 'theses', len(edges), 'genealogy as_of', gl.get('as_of'))


if __name__ == '__main__':
    main()
