#!/bin/bash
# 토픽 모형 11가지 (정식 LDA K=20 + 큰 실행 9 + 안정 토픽만) — 학맥 2026-11-22
cd "<PROJECT>/KCI_현대문학_2000-2026"
N="참고문헌_구축/학풍_학맥20261122"; REPO="$N/_genealogy_20261122"
while ! grep -q "DONE3" "$N/_rerun_chain2.log"; do sleep 20; done
for m in segment_k20 segment_k20_stable nmf km_bge_m3 km_e5_large km_ko_sroberta km_mpnet bt_bge_m3 bt_e5_large bt_ko_sroberta bt_mpnet; do
  echo "=== [$(date +%T)] $m 전체"; py -X utf8 수집도구/stat_review.py --prelim "$N/토픽모형/$m" --repo "$REPO" --out "$N/토픽모형/$m/검증" --label "$m (학맥 2026-11-22)" 2>&1 | tail -3
done
echo "=== [$(date +%T)] segment_k20 확정"; py -X utf8 수집도구/stat_review.py --prelim "$N/토픽모형/segment_k20" --repo "$REPO" --out "$N/토픽모형/segment_k20/검증_확정" --label "segment_k20 확정 (학맥 2026-11-22)" --edge-status 확정 2>&1 | tail -3
echo "=== [$(date +%T)] DONE4"
