#!/bin/bash
cd "<PROJECT>/KCI_현대문학_2000-2026"
N="참고문헌_구축/학풍_학맥20261122"; REPO="$N/_genealogy_20261122"
for v in all nlk_titles nlk_toc all_324 nlk_titles_324 nlk_toc_324; do
  echo "=== [$(date +%T)] 국중 $v 전체"; py -X utf8 수집도구/stat_review.py --prelim "$N/국중목차/$v" --repo "$REPO" --out "$N/국중목차/$v/검증" --label "국중 $v (학맥 2026-11-22)" 2>&1 | tail -3
  echo "=== [$(date +%T)] 국중 $v 확정"; py -X utf8 수집도구/stat_review.py --prelim "$N/국중목차/$v" --repo "$REPO" --out "$N/국중목차/$v/검증_확정" --label "국중 $v 확정 (학맥 2026-11-22)" --edge-status 확정 2>&1 | tail -3
done
echo "=== [$(date +%T)] DONE2"
