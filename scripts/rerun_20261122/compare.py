# -*- coding: utf-8 -*-
"""옛 결과(학맥 2026-11-10)와 새 결과(학맥 2026-11-22)를 나란히 놓는다. 실행: py -X utf8 compare.py <참고문헌_구축> <출력 md>"""
import sys, json
from pathlib import Path
G = Path(sys.argv[1]); N = G / '학풍_학맥20261122'; OLD = G / '학풍_현행학맥_20261114'
REL = [('advisor', '지도교수'), ('sibling', '동문'), ('uncle', '3촌'), ('grand', '조부모'), ('lineage_other', '같은 계보 기타'), ('same_uni_outside_lineage', '계보 밖 같은 대학')]
def load(p):
    p = Path(p); return json.load(open(p, encoding='utf-8')) if p.exists() else None
def sd(j, k): return j['main'][k]['std_effect']
def f2(x): return f'{x:+.2f}'
def p3(x): return '<0.001' if x < 0.0005 else f'{x:.3f}'
L = ['# 재분석 비교 — 학맥 2026-11-10 판 → 2026-11-22 판', '', '옛 = 학맥 2026-11-10 판(논문 초안의 수치), 새 = 연구자가 검수해 추가한 학맥 2026-11-22 판. 표준화 효과 = 계수 ÷ 잔차 표준편차.', '']
# 1 본 분석
for tier, lab in [('검증', '전체 간선'), ('검증_확정', '확정 간선만')]:
    o = load(OLD / 'bge_m3' / tier / 'stat_review.json'); n = load(N / 'bge_m3' / tier / 'stat_review.json')
    if not (o and n): continue
    L += [f'## 본 분석(bge-m3, KCI 논문 묶음) — {lab}', '', f"표본: 옛 교수 {o['n_professors']}명 × 논문 {o['n_theses']:,}편 → 새 교수 {n['n_professors']}명 × 논문 {n['n_theses']:,}편. 교수 사이 사제 관계 옛 {o['prof_prof_edges']['used']} → 새 {n['prof_prof_edges']['used']}건.", '',
          '| 관계 | 쌍 수 (옛 → 새) | 계수 (옛 → 새) | 표준화 효과 (옛 → 새) | t (옛 → 새) | Holm p (옛 → 새) | 순열 p (옛 → 새) |', '|---|---|---|---|---|---|---|']
    for k, kl in REL:
        a, b = o['main'][k], n['main'][k]
        L.append(f"| {kl} | {a['n_pairs']:,} → {b['n_pairs']:,} | {a['beta']:+.4f} → {b['beta']:+.4f} | {f2(a['std_effect'])} → {f2(b['std_effect'])} | {a['t_twoway']:.1f} → {b['t_twoway']:.1f} | {p3(a['p_holm'])} → {p3(b['p_holm'])} | {p3(a['perm_p'])} → {p3(b['perm_p'])} |")
    L.append('')
# 2 여덟 표현
NAMES = [('bge_m3', 'bge-m3'), ('e5_large', 'e5-large'), ('ko_sroberta', 'ko-sroberta'), ('mpnet_multilingual', 'MPNet'), ('bge_subset', 'bge-m3(본문 표본)'), ('tfidf', 'TF-IDF'), ('lsa300', 'LSA 300'), ('lda50', 'LDA 50')]
def table(title, items, oldf, newf, tier='검증'):
    out = [f'## {title}', '', '칸마다 「옛 → 새」. 조부모와 계보 밖 같은 대학은 괄호에 Holm p.', '', '| 표현 | 논문 수 | 지도교수 | 동문 | 3촌 | 조부모 | 같은 계보 기타 | 계보 밖 같은 대학 |', '|---|---|---|---|---|---|---|---|']
    for key, lab in items:
        o = load(oldf(key) / tier / 'stat_review.json'); n = load(newf(key) / tier / 'stat_review.json')
        if not n: out.append(f'| {lab} | (새 결과 없음) | | | | | | |'); continue
        def cell(k, withp=False):
            s = (f2(sd(o, k)) + ' → ' if o else '') + f2(sd(n, k))
            if withp: s += ' (' + (p3(o['main'][k]['p_holm']) + ' → ' if o else '') + p3(n['main'][k]['p_holm']) + ')'
            return s
        out.append(f"| {lab} | {(format(o['n_theses'], ',') + ' → ') if o else ''}{n['n_theses']:,} | {cell('advisor')} | {cell('sibling')} | {cell('uncle')} | {cell('grand', True)} | {cell('lineage_other')} | {cell('same_uni_outside_lineage', True)} |")
    return out + ['']
L += table('여덟 표현의 표준화 효과 — 전체 간선', NAMES, lambda k: OLD / k, lambda k: N / k)
L += table('여덟 표현의 표준화 효과 — 확정 간선만', NAMES, lambda k: OLD / k, lambda k: N / k, '검증_확정')
NLK = [('nlk_titles', '저작 제목'), ('nlk_toc', '저작 제목 + 목차'), ('all', 'KCI 논문 + 제목 + 목차'), ('nlk_titles_324', '저작 제목, 표본 교수 고정'), ('nlk_toc_324', '저작 제목 + 목차, 표본 교수 고정'), ('all_324', 'KCI 논문 + 제목 + 목차, 표본 교수 고정')]
L += table('국립중앙도서관 저서 제목·목차 묶음(bge-m3) — 전체 간선', NLK, lambda k: OLD / '국중목차' / k, lambda k: N / '국중목차' / k)
F = G / '학풍_토픽검토_20261120/정식_조각20/rep'; BR = G / '학풍_토픽_임베딩모형_20261122/큰실행/rep'
TM = [('segment_k20', 'LDA K=20(정식)'), ('segment_k20_stable', 'LDA K=20, 되풀이 토픽만'), ('nmf', 'NMF'), ('km_bge_m3', 'k-means · bge-m3'), ('km_e5_large', 'k-means · e5'), ('km_ko_sroberta', 'k-means · ko-sroberta'), ('km_mpnet', 'k-means · MPNet'),
      ('bt_bge_m3', 'BERTopic · bge-m3'), ('bt_e5_large', 'BERTopic · e5'), ('bt_ko_sroberta', 'BERTopic · ko-sroberta'), ('bt_mpnet', 'BERTopic · MPNet')]
L += table('토픽 모형 — 전체 간선', TM, lambda k: (F if k.startswith('segment') else BR) / k, lambda k: N / '토픽모형' / k)
# 3 시간 창
o = load(OLD / 'bge_m3/검증/stat_review.json'); n = load(N / 'bge_m3/검증/stat_review.json')
if o and n and o.get('windows') and n.get('windows'):
    L += ['## 전승 방향 — 지도교수 묶음을 학위 연도 기준 창으로 나눈 지도교수 효과(bge-m3)', '', '| 창 | 옛 | 새 |', '|---|---|---|']
    ow, nw = o['windows'], n['windows']
    def get(w, k):
        v = w.get(k) if isinstance(w, dict) else None
        if not v or 'advisor' not in v: return ''
        a = v['advisor']; ci = a.get('ci95_thesis_boot') or [float('nan')] * 2
        sib = v.get('sibling', {}).get('mean_resid'); uni = v.get('same_uni_outside_lineage', {}).get('mean_resid')
        return f"지도교수 {a['mean_resid']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] (쌍 {a['n_pairs']:,}) · 동문 {sib:+.4f} · 계보 밖 같은 대학 {uni:+.4f}"
    for k in (nw.keys() if isinstance(nw, dict) else []): L.append(f'| {k} | {get(ow, k)} | {get(nw, k)} |')
    L.append('')
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L[:60]))
