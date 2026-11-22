"""계보 판정 민감도 분석 (2026-10-31, 연구자 「돌려」)

형제·삼촌·조부·같은 계보 관계는 교수→교수 동일인 판정 간선(genealogy-library.json, 223건: 확정 139 · 추론 30 · 미확인 54)에 기댄다.
stat_review.py --edge-status 로 간선을 「확정만」·「확정+추론」으로 제한해 다시 추정하고, 전체 간선 결과(기존 검증/)와 비교표를 만든다.
지도교수 항(advisor)은 지도교수→논문 간선(도서관·RISS 서지)에서 오므로 이 제한의 영향을 받지 않는다 — 변하지 않아야 한다.

사용: py -X utf8 lineage_sensitivity.py --base <KCI_현대문학_2000-2026> --repo <<SITE>> [--only-report] [--boot 300 --perm 300]
출력: 참고문헌_구축\학풍_TFIDF_토픽\계보민감도_보고서_20261031.md + 계보민감도_20261031.json
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

TIERS = [('확정', '검증_확정'), ('확정,추론', '검증_확정추론'), ('', '검증')]
TIER_LAB = {'확정': '확정만 (139)', '확정,추론': '확정+추론 (169)', '': '전체 (223)'}
REL = ['advisor', 'sibling', 'uncle', 'grand', 'lineage_other', 'same_uni_outside_lineage', 'same_uni_x_lineage']
LAB = {'advisor': '지도교수', 'grand': '조부 교수', 'sibling': '형제 교수', 'uncle': '삼촌 교수', 'lineage_other': '같은 계보 기타', 'same_uni_outside_lineage': '계보 밖 같은 대학', 'same_uni_x_lineage': '같은 대학×계보'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--repo', required=True)
    ap.add_argument('--boot', type=int, default=300); ap.add_argument('--perm', type=int, default=300); ap.add_argument('--only-report', action='store_true')
    a = ap.parse_args(); base = Path(a.base); G = base / '참고문헌_구축'
    variants = [('bge (KCI만 전체, 본 분석)', G / '학풍_A기준_KCI만_전체'), ('bge_subset', G / '학풍_TFIDF_토픽/bge_subset'), ('tfidf', G / '학풍_TFIDF_토픽/tfidf'),
                ('lsa300', G / '학풍_TFIDF_토픽/lsa300'), ('lda50', G / '학풍_TFIDF_토픽/lda50')]
    t0 = time.time()
    if not a.only_report:
        for vl, vp in variants:
            for st, sub in TIERS[:2]:
                cmd = [sys.executable, '-X', 'utf8', str(base / '수집도구/stat_review.py'), '--prelim', str(vp), '--repo', a.repo, '--out', str(vp / sub), '--boot', str(a.boot), '--perm', str(a.perm),
                       '--label', f'{vl} · 교수 간선 {TIER_LAB[st]}', '--edge-status', st]
                print(f'[{time.time() - t0:7.0f}s] {vl} {st or "all"}', flush=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    R = {}
    for vl, vp in variants:
        R[vl] = {}
        for st, sub in TIERS:
            f = vp / sub / 'stat_review.json'
            if f.exists(): R[vl][st] = json.load(open(f, encoding='utf-8'))
    out_json = G / '학풍_TFIDF_토픽/계보민감도_20261031.json'
    json.dump({vl: {TIER_LAB[st]: {'prof_prof_edges': r.get('prof_prof_edges'), 'main': r['main']} for st, r in d.items()} for vl, d in R.items()}, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L = ['# 계보 판정 민감도 — 교수→교수 동일인 간선을 확정만 / 확정+추론 / 전체로 제한', '',
         f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 형제·삼촌·조부·같은 계보 관계를 만드는 교수→교수 간선(genealogy-library.json `student_identity.status`)을 세 단계로 제한해 stat_review.py 를 다시 돌렸다. '
         '지도교수 항은 지도교수→논문 간선(도서관·RISS)에서 오므로 세 단계에서 같아야 한다(확인용). 괄호 안은 양방향 클러스터 t, `*` = Holm p < 0.05, n = 쌍 수. 기준 = 다른 대학·계보 밖.', '']
    for vl, d in R.items():
        if not d: continue
        L += [f'## {vl}', '']
        hdr = [TIER_LAB[st] for st in TIERS if st in d]
        L += ['| 관계 | ' + ' | '.join(hdr) + ' |', '|---|' + '---|' * len(hdr)]
        for rel in REL:
            cells = []
            for st in TIERS:
                if st[0] not in d: continue
                m = d[st[0]]['main'].get(rel)
                cells.append('—' if not m else f"{m['beta']:+.4f} ({m['t_twoway']:.1f}){'*' if m['p_holm'] < 0.05 else ''} n={m['n_pairs']}")
            L.append(f'| {LAB[rel]} | ' + ' | '.join(cells) + ' |')
        L.append('')
        L += ['표준화 효과(계수 / 잔차 SD):', '', '| 관계 | ' + ' | '.join(hdr) + ' |', '|---|' + '---|' * len(hdr)]
        for rel in REL:
            cells = []
            for st in TIERS:
                if st[0] not in d: continue
                m = d[st[0]]['main'].get(rel); cells.append('—' if not m else f"{m['std_effect']:+.2f}")
            L.append(f'| {LAB[rel]} | ' + ' | '.join(cells) + ' |')
        pp = {st: d[st]['prof_prof_edges'] for st, _ in TIERS if st in d and d[st].get('prof_prof_edges')}
        if pp: L += ['', '교수→교수 간선 사용: ' + ' · '.join(f"{TIER_LAB[st]} → {v['used']}건, 지도교수 있는 교수 {v['professors_with_advisor']}명" for st, v in pp.items())]
        L.append('')
    L += ['## 읽는 법', '',
          '- 지도교수 계수가 세 열에서 같으면 제한이 바르게 걸렸다는 뜻이다(지도교수→논문 간선은 건드리지 않음).',
          '- 확정만 열에서 형제·삼촌·조부 계수가 전체 열과 같은 부호·같은 자릿수이고 Holm 유의가 유지되면 계보 효과는 미확인 54건·추론 30건에 기대지 않는다. 쌍 수(n)가 줄어 t 는 작아질 수 있다.',
          '- 간선을 빼면 그 제자 교수는 「계보 밖」으로 가므로 기준 집단·같은 대학 항에 섞인다. 「계보 밖 같은 대학」이 확정만 열에서 조금 커지면 그 때문이다(미확인 간선이 실제 계보라는 간접 증거).',
          '- 전체 열은 기존 `검증/` 결과를 그대로 읽은 것이다.']
    (G / '학풍_TFIDF_토픽/계보민감도_보고서_20261031.md').write_text('\n'.join(L), encoding='utf-8'); print('report written', G / '학풍_TFIDF_토픽/계보민감도_보고서_20261031.md')


if __name__ == '__main__':
    main()
