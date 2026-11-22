#!/bin/bash
# 역산(주제 묶음) — 학맥 2026-11-22. 검정이 모두 끝난 뒤 실행
N="<PROJECT>/KCI_현대문학_2000-2026/참고문헌_구축/학풍_학맥20261122"
while ! grep -q "DONE4" "$N/_rerun_chain3.log"; do sleep 30; done
cd "<HOME>/AppData/Local/Temp/claude/660ffe1d11cd8ac7/4aa2ad42-eda4-41ca-9651-bbdba845ca58/scratchpad/bt"
echo "=== [$(date +%T)] 역산"
./venv/Scripts/python.exe -X utf8 rev3_run.py big "$N/_genealogy_20261122" "$N/_genealogy_20261122" "<PROJECT>/kmmocr/색인_v3/text_sp" "$N" rev3_out "$N/토픽모형/segment_k20/vectors_meta.json" 2>&1 | grep "^\[" | cut -c1-700
mkdir -p "$N/역산"; cp rev3_out/reverse_refined.json "$N/역산/"
echo "=== [$(date +%T)] DONE5"
