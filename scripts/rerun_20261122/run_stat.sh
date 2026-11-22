#!/bin/bash
# 표현 8개 × 간선 단계 3개 검정 (학맥 2026-11-22)
cd "<PROJECT>/KCI_현대문학_2000-2026"
N="참고문헌_구축/학풍_학맥20261122"; REPO="$N/_genealogy_20261122"
for m in bge_m3 e5_large ko_sroberta mpnet_multilingual bge_subset tfidf lsa300 lda50; do
  echo "=== [$(date +%T)] $m 전체"
  py -X utf8 수집도구/stat_review.py --prelim "$N/$m" --repo "$REPO" --out "$N/$m/검증" --label "$m (학맥 2026-11-22)" 2>&1 | tail -3
  echo "=== [$(date +%T)] $m 확정"
  py -X utf8 수집도구/stat_review.py --prelim "$N/$m" --repo "$REPO" --out "$N/$m/검증_확정" --label "$m 확정 (학맥 2026-11-22)" --edge-status 확정 2>&1 | tail -3
  echo "=== [$(date +%T)] $m 확정추론"
  py -X utf8 수집도구/stat_review.py --prelim "$N/$m" --repo "$REPO" --out "$N/$m/검증_확정추론" --label "$m 확정·추론 (학맥 2026-11-22)" --edge-status 확정,추론 2>&1 | tail -3
done
echo "=== [$(date +%T)] DONE"
