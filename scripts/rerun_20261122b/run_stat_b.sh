#!/bin/bash
# 학맥 2026-11-22 밤 판(a6bed358): 표본·벡터는 그대로(학풍_학맥20261122), 계보 관계만 새 판으로. 결과는 학풍_학맥20261122b 에.
cd "<PROJECT>/KCI_현대문학_2000-2026"
N="참고문헌_구축/학풍_학맥20261122"; N2="참고문헌_구축/학풍_학맥20261122b"; REPO="$N2/_genealogy"
run() { # prelim-rel out-rel label [edge-status]
  echo "=== [$(date +%T)] $2"
  if [ -n "$4" ]; then py -X utf8 수집도구/stat_review.py --prelim "$N/$1" --repo "$REPO" --out "$N2/$2" --label "$3" --edge-status "$4" 2>&1 | tail -2
  else py -X utf8 수집도구/stat_review.py --prelim "$N/$1" --repo "$REPO" --out "$N2/$2" --label "$3" 2>&1 | tail -2; fi
}
run bge_m3 bge_m3/검증 "bge_m3 (학맥 2026-11-22 밤)"
for m in bge_m3 e5_large ko_sroberta mpnet_multilingual bge_subset tfidf lsa300 lda50; do
  run $m $m/검증_확정 "$m 확정 (학맥 2026-11-22 밤)" 확정
  run $m $m/검증_확정추론 "$m 확정·추론 (학맥 2026-11-22 밤)" 확정,추론
done
for v in all nlk_titles nlk_toc all_324 nlk_titles_324 nlk_toc_324; do run 국중목차/$v 국중목차/$v/검증_확정 "국중 $v 확정 (학맥 2026-11-22 밤)" 확정; done
run 토픽모형/segment_k20 토픽모형/segment_k20/검증_확정 "segment_k20 확정 (학맥 2026-11-22 밤)" 확정
echo "=== [$(date +%T)] DONE"
