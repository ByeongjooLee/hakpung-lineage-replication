# -*- coding: utf-8 -*-
"""비교 보고서 뒷부분: 식별·코호트·제자 궤적·토픽 η². 실행: py -X utf8 compare2.py <참고문헌_구축> <md(덧붙임)>"""
import sys, json, csv
from pathlib import Path
G = Path(sys.argv[1]); N = G / '학풍_학맥20261122'; O = G / '학풍_현행학맥_20261114'
J = lambda p: json.load(open(p, encoding='utf-8'))
L = []
a = J(G / '그림_20261121/그림4_summary.json'); b = J(G / '그림_20261122/그림4_summary.json')
L += ['## 지도교수 식별(같은 대학 교수 가운데 지도교수 묶음의 순위, bge-m3)', '', '| | 옛 | 새 |', '|---|---|---|',
      f"| 대상 논문 | {a['n_same_uni']:,} | {b['n_same_uni']:,} |", f"| 1위로 맞힌 비율 | {a['top1_same_uni']*100:.1f}% | {b['top1_same_uni']*100:.1f}% |", f"| 우연 기대 | {a['chance_top1_same_uni']*100:.1f}% | {b['chance_top1_same_uni']*100:.1f}% |",
      f"| 전체 교수 가운데 1위 | {a['top1_all']*100:.1f}% (교수 {a['n_prof_all']}) | {b['top1_all']*100:.1f}% (교수 {b['n_prof_all']}) |", '']
co, cn = J(O / '코호트/cohort_regression.json'), J(N / '코호트/cohort_regression.json')
def cg(d, k):
    v = d.get(k) if isinstance(d, dict) else None
    if v is None and isinstance(d, dict):
        for kk in d:
            if isinstance(d[kk], dict) and k in d[kk]: v = d[kk][k]; break
    if isinstance(v, dict): return f"{v.get('beta', float('nan')):+.4f} (t {v.get('t', float('nan')):.1f}, 쌍 {v.get('n', v.get('n_pairs', 0)):,})"
    if isinstance(v, list): return f"{v[0]:+.4f} (t {v[1]:.1f}, 쌍 {v[2]:,})"
    return ''
keys = ['advisor_≤10', 'advisor_11–20', 'advisor_21–30', 'advisor_>30', 'sibling_≤5', 'sibling_6–15', 'sibling_>15', 'uncle', 'grand', 'lineage_other', 'sameuni_≤5', 'sameuni_6–15', 'sameuni_>15']
lab = {'advisor': '지도교수(교수의 학위 뒤 연수', 'sibling': '동문(학위 연도 차', 'sameuni': '계보 밖 같은 대학(학위 연도 차'}
L += ['## 코호트 분리(bge-m3)', '', '| 항 | 옛 | 새 |', '|---|---|---|']
for k in keys:
    h, _, t = k.partition('_'); name = (lab[h] + ' ' + t + ')') if h in lab else {'uncle': '3촌', 'grand': '조부모', 'lineage_other': '같은 계보 기타'}[k]
    L.append(f'| {name} | {cg(co, k)} | {cg(cn, k)} |')
L.append('')
to, tn = J(O / '제자궤적/trajectory.json'), J(N / '제자궤적/trajectory.json')
def bins(t):
    for k in ('bins', 'bin_means', 'by_bin'):
        if k in t: return t[k]
    return {}
bo, bn = bins(to), bins(tn)
L += ['## 제자 궤적(학위 뒤 본인 KCI 논문과 교수 묶음의 잔차 유사도)', '', f"제자 수·논문 수: 옛 {to.get('n_students_with_articles', '')}명·{to.get('n_articles', '')}편 → 새 {tn.get('n_students_with_articles', '')}명·{tn.get('n_articles', '')}편.", '', '| 학위 뒤 연차 | 지도교수 (옛 → 새) | 지도교수의 학위 전 10년 주제 (옛 → 새) | 재직 대학 동료 (옛 → 새) |', '|---|---|---|---|']
def gv(d, k, f):
    v = d.get(k, {}).get(f) if isinstance(d, dict) else None
    if isinstance(v, dict): v = v.get('mean')
    return f'{v:+.4f}' if isinstance(v, (int, float)) else ''
for k in (bn if isinstance(bn, dict) else {}):
    L.append(f"| {k} | {gv(bo, k, 'advisor_resid')} → {gv(bn, k, 'advisor_resid')} | {gv(bo, k, 'advisor_pre_resid')} → {gv(bn, k, 'advisor_pre_resid')} | {gv(bo, k, 'work_colleague_resid')} → {gv(bn, k, 'work_colleague_resid')} |")
L.append('')
eo, en = J(O / '토픽/eta2_k50.json'), J(N / '토픽/eta2_k50.json')
L += ['## 토픽 비중의 집단 간 분산(η², LDA K=50) — 순열 귀무 평균을 뺀 초과', '', '| 집단 | 옛: η² (귀무) · 초과 · 논문·집단 수 | 새 |', '|---|---|---|']
for k, kl in [('advisor', '지도교수'), ('lineage_root', '계보 뿌리'), ('university', '학위 대학'), ('university_x_lineage', '대학×계보')]:
    def f(e):
        v = e[k]; return f"{v['eta2']:.3f} ({v['null_mean']:.3f}) · +{v['eta2'] - v['null_mean']:.3f} · {v['n_theses']:,}편·{v['n_groups']}"
    L.append(f'| {kl} | {f(eo)} | {f(en)} |')
for k, kl in [('university_beyond_lineage', '계보 위에 대학을 더한 증분'), ('lineage_beyond_university', '대학 위에 계보를 더한 증분')]:
    def f(e):
        v = e.get(k)
        return f"{v['increment']:.3f} (귀무 {v['null_mean']:.3f}) · +{v['increment'] - v['null_mean']:.3f}" if v else ''
    L.append(f'| {kl} | {f(eo)} | {f(en)} |')
L.append('')
open(sys.argv[2], 'a', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L))
