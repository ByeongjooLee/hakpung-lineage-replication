#!/bin/bash
# 검정이 끝난 뒤: 코호트 · 제자 궤적 · 계보×토픽 (학맥 2026-11-22)
cd "<PROJECT>/KCI_현대문학_2000-2026"
N="참고문헌_구축/학풍_학맥20261122"; REPO="$N/_genealogy_20261122"
while ! grep -q "DONE" "$N/_rerun.log"; do sleep 20; done
while ! grep -q "DONE2" "$N/_rerun_nlk.log"; do sleep 20; done
echo "=== [$(date +%T)] 코호트"
py -X utf8 수집도구/cohort_vs_lineage.py --prelim "$N/bge_m3" --repo "$REPO" --out "$N/코호트" --phd-years 참고문헌_구축/저자전거/prof_phd_years.csv 2>&1 | tail -5
echo "=== [$(date +%T)] 제자 궤적"
py -X utf8 "$N/_scripts/student_trajectory_fromcsv.py" --prelim "$N/bge_m3" --repo "$REPO" --base . --out "$N/제자궤적" --docs-from "$N/제자궤적/_docs_from.csv" 2>&1 | tail -8
echo "=== [$(date +%T)] 계보×토픽"
py -X utf8 "$N/_scripts/topic_lineage_recompute.py" --lda "$N/lda50" --repo "$REPO" --out "$N/토픽" 2>&1 | tail -5
echo "=== [$(date +%T)] DONE3"
