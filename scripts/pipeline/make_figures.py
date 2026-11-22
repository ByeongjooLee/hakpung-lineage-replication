"""학풍·계보 논문 그림 (2026-11-01, 연구자 「그림 시작」)

입력 = 공개 파생표(기본 공개_파생표_20261101) + 학풍_A기준_KCI만_전체/edges_similarity.csv(그림 4 순위) + 학풍_TFIDF_토픽/{lineage_topics_k50,topics_k50}.csv(그림 7).
출력(--out, 기본 참고문헌_구축/그림_YYYYMMDD): 그림마다 SVG·PNG(300dpi)와 표 쌍둥이 CSV, 그림_캡션.md.
  그림 1  관계별 표준화 효과 포레스트 — 표현 7개 × 관계 5개, 교수 부트스트랩 95% 구간, 확정 간선만 추정치를 빈 마름모로 겹침
  그림 2  전승 방향 — 지도교수 묶음을 학위 연도 기준 창으로 나눈 평균 잔차(학위 전 10년 창 = 100 지수), bge 강조 + 95% 구간, 다른 표현은 회색
  그림 3  제자 궤적 — 학위 후 연차 구간별 지도교수 각인 / 물려받은 학위 전 주제 / 재직 대학 동료 (95% 구간)
  그림 4  지도교수 식별 — 같은 대학 교수 중 지도교수 묶음의 유사도 순위 분포 vs 우연 기대
  그림 7  계보×토픽 — 계보 뿌리별 LDA 토픽 비중 히트맵(상위 계보 10, 특이 토픽 합집합)
색은 dataviz 기본 범주형(1 파랑·2 주황·3 청록)과 파랑 순차 램프만 쓴다. 글꼴 Noto Sans KR.
실행: py -X utf8 make_figures.py --base <KCI_현대문학_2000-2026> [--tables 공개_파생표_20261101] [--out 그림_20261101]
"""
import argparse, csv, json, time, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
REPS = [('bge', 'bge-m3', 'emb'), ('e5_large', 'e5-large', 'emb'), ('ko_sroberta', 'ko-sroberta', 'emb'), ('mpnet_multilingual', 'MPNet', 'emb'), ('tfidf', 'TF-IDF', 'lex'), ('lsa300', 'LSA-300', 'lex'), ('lda50', 'LDA-50', 'lex')]
RELS = [('advisor', '지도교수'), ('sibling', '형제 교수'), ('uncle', '삼촌 교수'), ('grand', '조부 교수'), ('same_uni_outside_lineage', '계보 밖 같은 대학')]


def setup_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ['Malgun Gothic', 'NanumGothic', 'Noto Sans KR']:   # Noto Sans KR 은 이 PC 에 Thin(100)만 있어 글자가 가늘다
        if cand in names: plt.rcParams['font.family'] = cand; break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2,
                         'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def despine(ax, keep=('left', 'bottom')):
    for s in ax.spines:
        ax.spines[s].set_visible(s in keep)


def save(fig, out, name):
    fig.savefig(out / f'{name}.svg', bbox_inches='tight'); fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def wcsv(path, cols, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def fig1(T, out):
    reg = list(csv.DictReader(open(T / 'regression_results.csv', encoding='utf-8-sig')))
    R = {(r['representation'], r['edge_tier'], r['relation']): r for r in reg}
    rows = []
    fig, axes = plt.subplots(1, len(RELS), figsize=(11, 4.6), sharey=True)
    for ax, (rel, rlab) in zip(axes, RELS):
        ys = []
        for i, (rep, lab, fam) in enumerate(REPS):
            y = len(REPS) - 1 - i; ys.append(y); col = C1 if fam == 'emb' else C2
            m = R.get((rep, 'all', rel)); c = R.get((rep, 'confirmed', rel))
            if not m: continue
            sd = float(m['sd_resid']); b = float(m['std_effect']); lo = float(m['ci95_boot_prof_lo']) / sd; hi = float(m['ci95_boot_prof_hi']) / sd
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.plot(b, y, 'o', ms=7, color=col, mec=SURF, mew=1.5, zorder=4)
            rows.append({'relation': rel, 'relation_label': rlab, 'representation': rep, 'std_effect_all': f'{b:.3f}', 'ci95_lo': f'{lo:.3f}', 'ci95_hi': f'{hi:.3f}', 'p_holm_all': m['p_holm'], 'n_pairs': m['n_pairs']})
            if c:
                bc = float(c['std_effect']); ax.plot(bc, y - 0.28, 'D', ms=5, mfc=SURF, mec=col, mew=1.3, zorder=5)
                rows[-1].update({'std_effect_confirmed': f'{bc:.3f}', 'p_holm_confirmed': c['p_holm']})
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1); ax.set_title(rlab, pad=6)
        ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab, _ in REPS]); ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', zorder=0); despine(ax, keep=('bottom',)); ax.set_xlim(-0.25, 1.55); ax.set_xticks([0, 0.5, 1.0, 1.5])
        for lbl in ax.get_yticklabels(): lbl.set_color(INK2)
    axes[0].set_ylabel('')
    fig.supxlabel('표준화 효과 (계수 ÷ 잔차 표준편차)', color=INK2, fontsize=9, y=-0.02)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C1, marker='o', lw=2, ms=7, mec=SURF, label='임베딩 표현 (전체 간선)'), Line2D([], [], color=C2, marker='o', lw=2, ms=7, mec=SURF, label='어휘 표현 (전체 간선)'),
               Line2D([], [], color=INK2, marker='D', lw=0, ms=5, mfc=SURF, mec=INK2, label='확정 간선만 (점 추정)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.suptitle('관계별 주제 유사도 효과 — 일곱 표현, 교수 부트스트랩 95% 구간', x=0.01, ha='left', y=1.08, fontsize=11, fontweight='bold')
    fig.tight_layout(); save(fig, out, '그림1_관계별_효과_포레스트')
    wcsv(out / '그림1_data.csv', ['relation', 'relation_label', 'representation', 'std_effect_all', 'ci95_lo', 'ci95_hi', 'p_holm_all', 'n_pairs', 'std_effect_confirmed', 'p_holm_confirmed'], rows)


def fig2(T, out):
    W = list(csv.DictReader(open(T / 'window_results.csv', encoding='utf-8-sig')))
    order = [('위약: 학위 11년 전 이전', '-30…-11\n(위약)'), ('학위 전 10년', '-10…-1'), ('학위 전후', '-3…+3'), ('학위 후 10년', '+1…+10'), ('위약: 학위 11년 뒤 이후', '+11…+30\n(위약)')]   # Malgun Gothic 에 U+2212 가 없어 하이픈
    def val(rep, key, field='mean_resid'):
        r = next((x for x in W if x['representation'] == rep and x['edge_tier'] == 'all' and x['relation'] == 'advisor' and x['window'].startswith(key)), None)
        return r
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); x = np.arange(len(order)); rows = []
    for rep, lab, fam in REPS:
        vals = [val(rep, k) for k, _ in order]
        if any(v is None or v['mean_resid'] == '' for v in vals): continue
        base = float(val(rep, '학위 전 10년')['mean_resid']); idx = [float(v['mean_resid']) / base * 100 for v in vals]
        for (k, _), v, ix in zip(order, vals, idx):
            rows.append({'representation': rep, 'window': v['window'], 'mean_resid': v['mean_resid'], 'index_vs_pre10': f'{ix:.1f}', 'n_pairs': v['n_pairs'], 'n_theses': v['n_theses'], 'ci95_lo': v['ci95_thesis_boot_lo'], 'ci95_hi': v['ci95_thesis_boot_hi']})
        if rep == 'bge':
            lo = [float(v['ci95_thesis_boot_lo']) / base * 100 for v in vals]; hi = [float(v['ci95_thesis_boot_hi']) / base * 100 for v in vals]
            ax.fill_between(x, lo, hi, color=C1, alpha=0.10, lw=0, zorder=1)
            ax.plot(x, idx, color=C1, lw=2.2, marker='o', ms=8, mec=SURF, mew=1.5, zorder=4, label='bge-m3 (본 분석, 띠 = 논문 부트스트랩 95%)')
        else:
            ax.plot(x, idx, color=MUTED, lw=1.2, alpha=0.7, zorder=2, label='다른 표현 6개' if rep == 'e5_large' else None)
    ax.axhline(100, color=AXIS, lw=0.8, zorder=0); ax.set_xticks(x); ax.set_xticklabels([l for _, l in order])
    ax.set_ylabel('지도교수 묶음과의 평균 잔차 유사도\n(학위 전 10년 창 = 100)', color=INK2); ax.set_xlabel('지도교수 논문의 발표 시점 (제자 학위 연도 기준, 년)', color=INK2)
    ax.grid(axis='y', zorder=0); despine(ax); ax.legend(frameon=False, loc='lower left', fontsize=8.5)
    ax.set_title('전승 방향 — 제자 논문은 지도교수의 학위 전 연구와 가장 닮는다', pad=10)
    fig.tight_layout(); save(fig, out, '그림2_전승방향_창별')
    wcsv(out / '그림2_data.csv', ['representation', 'window', 'mean_resid', 'index_vs_pre10', 'n_pairs', 'n_theses', 'ci95_lo', 'ci95_hi'], rows)


def fig3(T, out):
    B = list(csv.DictReader(open(T / 'trajectory_bins.csv', encoding='utf-8-sig')))
    bins = ['학위 전(−5…−1)', '0', '1–3', '4–6', '7–10', '11–15', '16+']
    series = [('advisor_resid', '지도교수 묶음 (각인)', C1), ('advisor_pre_resid', '지도교수의 학위 전 10년 주제', C2), ('work_colleague_resid', '재직 대학 동료 교수', C3)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); x = np.arange(len(bins)); rows = []
    for k, (meas, lab, col) in enumerate(series):
        m = {r['bin']: r for r in B if r['measure'] == meas}
        xs = [i for i, b in enumerate(bins) if b in m and m[b]['mean']]; ys = [float(m[bins[i]]['mean']) for i in xs]
        lo = [float(m[bins[i]]['ci95_lo']) for i in xs]; hi = [float(m[bins[i]]['ci95_hi']) for i in xs]
        off = (k - 1) * 0.08
        ax.plot(np.array(xs) + off, ys, color=col, lw=2, marker='o', ms=7, mec=SURF, mew=1.5, zorder=3 + k, label=lab)
        for xi, l, h in zip(xs, lo, hi): ax.plot([xi + off, xi + off], [l, h], color=col, lw=1.2, alpha=0.8, zorder=2)
        for i in xs: rows.append({'measure': meas, 'label': lab, 'bin': bins[i], 'mean': m[bins[i]]['mean'], 'ci95_lo': m[bins[i]]['ci95_lo'], 'ci95_hi': m[bins[i]]['ci95_hi'], 'n_articles': m[bins[i]]['n_articles'], 'n_students': m[bins[i]]['n_students']})
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0); ax.set_xticks(x); ax.set_xticklabels([b.replace('−', '-') for b in bins]); ax.set_xlabel('학위 후 연차 (제자 본인의 KCI 논문 발표 시점)', color=INK2)
    ax.set_ylabel('교수 묶음과의 평균 잔차 유사도 (bge-m3)', color=INK2); ax.grid(axis='y', zorder=0); despine(ax); ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.set_title('제자 궤적 — 지도교수 각인은 남고 재직 대학 동료 효과는 거의 없다', pad=10)
    fig.tight_layout(); save(fig, out, '그림3_제자궤적')
    wcsv(out / '그림3_data.csv', ['measure', 'label', 'bin', 'mean', 'ci95_lo', 'ci95_hi', 'n_articles', 'n_students'], rows)


def fig4(E, out):
    rows = list(csv.DictReader(open(E, encoding='utf-8-sig')))
    rk = [(int(r['emb_rank_same_uni']), int(r['n_prof_same_uni'])) for r in rows if r['emb_rank_same_uni'] and r['n_prof_same_uni'] and int(r['n_prof_same_uni']) >= 2]
    labels = ['1위', '2위', '3위', '4위', '5위 이하']
    def binof(k): return min(k, 5) - 1
    obs = np.zeros(5); exp = np.zeros(5)
    for k, n in rk:
        obs[binof(k)] += 1
        for j in range(1, n + 1): exp[binof(j)] += 1 / n
    obs /= len(rk); exp /= len(rk)
    all_rank = [int(r['emb_rank_all']) for r in rows if r['emb_rank_all']]; top1_all = np.mean([k == 1 for k in all_rank]); n_all = int(rows[0]['n_prof_all'])
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); x = np.arange(5); w = 0.34
    ax.bar(x, obs * 100, width=0.5, color=C1, zorder=3, label='관측 (지도교수 묶음의 실제 순위)')
    ax.hlines(exp * 100, x - 0.3, x + 0.3, color=INK2, lw=2.2, zorder=4, label='우연 기대 (같은 대학 교수 수의 역수)')
    for i in range(5): ax.text(x[i], max(obs[i], exp[i]) * 100 + 1, f'{obs[i] * 100:.0f}%', ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('박사논문 비율 (%)', color=INK2); ax.set_xlabel('같은 대학 교수 묶음 가운데 지도교수 묶음의 유사도 순위 (짧은 가로선 = 우연 기대)', color=INK2)
    ax.grid(axis='y', zorder=0); despine(ax); ax.legend(frameon=False, fontsize=8.5)
    ax.set_title(f'지도교수 식별 — 박사논문 {len(rk):,}편, 같은 대학 교수 2명 이상인 경우 (bge-m3)', pad=10)
    fig.tight_layout(); save(fig, out, '그림4_지도교수_순위')
    wcsv(out / '그림4_data.csv', ['rank_bin', 'observed_share', 'chance_share', 'n_theses'], [{'rank_bin': l, 'observed_share': f'{o:.4f}', 'chance_share': f'{e:.4f}', 'n_theses': len(rk)} for l, o, e in zip(labels, obs, exp)])
    return {'n_same_uni': len(rk), 'top1_same_uni': float(obs[0]), 'chance_top1_same_uni': float(exp[0]), 'top1_all': float(top1_all), 'n_prof_all': n_all}


def fig7(L, TOP, out, n_lineages=10):
    rows = list(csv.DictReader(open(L, encoding='utf-8-sig'))); topics = {int(r['topic']): r for r in csv.DictReader(open(TOP, encoding='utf-8-sig'))}
    rows = [r for r in rows if r['group'] not in ('전체', '기타')][:n_lineages]
    K = 50; M = np.array([[float(r[f't{k}']) for k in range(K)] for r in rows])
    # 토픽 선택: 계보마다 특이 토픽(top_distinct_topics) 앞 2개 + 비중 상위 2개의 합집합, 전체 비중 순으로 정렬
    sel = set()
    for r in rows:
        for tok in r['top_distinct_topics'].split()[:2]: sel.add(int(tok.split('×')[0][1:]))
        sel |= set(np.argsort(-np.array([float(r[f't{k}']) for k in range(K)]))[:2].tolist())
    sel = sorted(sel, key=lambda k: -M[:, k].mean())[:14]
    sub = M[:, sel] * 100
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list('seq', [SURF] + SEQ); norm = Normalize(0, max(10, sub.max()))
    im = ax.imshow(sub, cmap=cmap, norm=norm, aspect='auto')
    # 2px 표면 간격: 셀 사이 흰 선
    for i in range(1, sub.shape[0]): ax.axhline(i - 0.5, color=SURF, lw=2)
    for j in range(1, sub.shape[1]): ax.axvline(j - 0.5, color=SURF, lw=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['group']} ({r['n_theses']})" for r in rows]); ax.tick_params(length=0)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels([f"t{k} " + '·'.join(topics[k]['top_words'].split()[:3]) for k in sel], fontsize=8, rotation=35, ha='right', rotation_mode='anchor')
    for s in ax.spines: ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('계보 안 박사논문의 평균 토픽 비중 (%)', color=INK2); cb.outline.set_visible(False)
    ax.set_title(f'계보 뿌리별 토픽 분포 — LDA 50 토픽 중 특이·상위 토픽 {len(sel)}개 (괄호 = 논문 수)', pad=10)
    fig.tight_layout(); save(fig, out, '그림7_계보_토픽_히트맵')
    out_rows = []
    for r, m in zip(rows, sub):
        out_rows.append({'lineage_root': r['group'], 'n_theses': r['n_theses'], **{f't{k}': f'{v:.2f}' for k, v in zip(sel, m)}})
    wcsv(out / '그림7_data.csv', ['lineage_root', 'n_theses'] + [f't{k}' for k in sel], out_rows)
    wcsv(out / '그림7_topics.csv', ['topic', 'top_words'], [{'topic': f't{k}', 'top_words': topics[k]['top_words']} for k in sel])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True); ap.add_argument('--tables', default='공개_파생표_20261101'); ap.add_argument('--out')
    a = ap.parse_args(); base = Path(a.base); G = base / '참고문헌_구축'; T = G / a.tables
    out = Path(a.out) if a.out else G / ('그림_' + time.strftime('%Y%m%d')); out.mkdir(parents=True, exist_ok=True)
    setup_font()
    fig1(T, out); fig2(T, out); fig3(T, out)
    f4 = fig4(G / '학풍_A기준_KCI만_전체/edges_similarity.csv', out)
    fig7(G / '학풍_TFIDF_토픽/lineage_topics_k50.csv', G / '학풍_TFIDF_토픽/topics_k50.csv', out)
    cap = ['# 그림 캡션과 출처', '', f'생성 {time.strftime("%Y-%m-%d %H:%M")}. 입력 = `{a.tables}`(regression_results·window_results·trajectory_bins), `학풍_A기준_KCI만_전체/edges_similarity.csv`, `학풍_TFIDF_토픽/lineage_topics_k50.csv`. 그림마다 같은 이름의 `_data.csv` 가 표 쌍둥이다. 색: 파랑 = 임베딩 표현/본 분석, 주황 = 어휘 표현/두 번째 계열, 청록 = 세 번째 계열; 히트맵은 파랑 단일 순차 램프.', '',
           '**그림 1. 관계별 주제 유사도 효과.** 제자 박사논문 × 교수 묶음의 잔차 유사도를 관계 더미로 회귀한 계수를 잔차 표준편차로 나눈 표준화 효과. 가로선 = 교수 클러스터 부트스트랩 300회 95% 구간, 채운 점 = 전체 간선(확정+추론+미확인), 빈 마름모 = 확정 간선만으로 계보를 만든 추정치(지도교수 항은 같음). 일곱 표현(bge-m3·e5-large·ko-sroberta·MPNet 임베딩, TF-IDF·LSA·LDA 어휘)에서 지도교수 > 형제 ≳ 삼촌 > 조부 의 순서와 「계보 밖 같은 대학 ≈ 0」이 재현된다. 조부 항은 확정만 추정치가 작아진다. 출처 regression_results.csv.', '',
           '**그림 2. 전승 방향.** 지도교수의 KCI 논문을 제자 학위 연도 기준 창(−30…−11 위약, −10…−1, −3…+3, +1…+10, +11…+30 위약)으로 나눠 창별 묶음과 제자 논문의 평균 잔차 유사도를 「학위 전 10년 창 = 100」으로 지수화. bge-m3(파랑, 띠 = 논문 부트스트랩 95%)와 다른 여섯 표현(회색). 학위 전 창이 가장 높고 학위 11년 뒤 창이 가장 낮다 → 제자가 지도교수의 기존 연구를 이어받는 방향. 출처 window_results.csv.', '',
           '**그림 3. 제자 궤적.** 학위자 896명의 본인 KCI 논문 9,834편을 학위 후 연차 구간으로 나눠 지도교수 묶음(파랑), 지도교수의 학위 전 10년 주제(주황), 재직 대학 동료 교수 묶음(청록)과의 평균 잔차 유사도(bge-m3, 세로선 = 95% 구간). 지도교수 각인은 16년 뒤에도 남고 물려받은 옛 주제는 절반으로 줄며 재직 대학 동료 효과는 0 근처. 출처 trajectory_bins.csv.', '',
           f"**그림 4. 지도교수 식별.** 같은 대학 교수 묶음이 2개 이상인 박사논문 {f4['n_same_uni']:,}편에서 지도교수 묶음이 유사도 몇 위인지(파랑)와 우연 기대(빈 막대, 같은 대학 교수 수의 역수). 1위 {f4['top1_same_uni'] * 100:.1f}% vs 우연 {f4['chance_top1_same_uni'] * 100:.1f}%. 전체 교수 {f4['n_prof_all']}명 중 1위는 {f4['top1_all'] * 100:.1f}%. bge-m3 원 코사인 기준. 출처 edges_similarity.csv.", '',
           '**그림 7. 계보 뿌리별 토픽 분포.** 박사논문 본문 LDA(K=50) 토픽 비중을 계보 뿌리(논문 수 상위 10개)별로 평균한 값. 열 = 계보마다 특이 토픽 2개·상위 토픽 2개의 합집합(최대 14개), 열 머리에 토픽 상위 낱말 3개. 우연 기대선이 있는 그림 4 와 달리 비교 기준은 전체 평균이 아니라 계보 간 대조다. 출처 lineage_topics_k50.csv·topics_k50.csv.', '',
           '그림 5(교수×교수 블록 히트맵)와 6 은 만들지 않았다(조건부·불필요). 모든 수치는 수집 자료 안의 연결만 센 값이다.']
    (out / '그림_캡션.md').write_text('\n'.join(cap), encoding='utf-8')
    json.dump(f4, open(out / '그림4_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('done', out, f4)


if __name__ == '__main__':
    main()
