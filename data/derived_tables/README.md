# 학풍(지도교수 계보) 분석 공개 파생표 — 2026-11-22

한국 현대문학 박사학위논문과 지도교수의 KCI 논문 묶음 사이의 주제 유사도를, 지도교수·계보·대학 관계별로 비교한 분석의 재현 자료다. 원문·초록·임베딩 벡터는 넣지 않고, 쌍별 유사도(원·잔차)와 토픽 비중, 관계 표지, 결과표, 그리고 원자료를 되찾을 공개 식별자(RISS control_no, KCI ART id·cretId, 국중 KAC·ISNI)만 담았다.

표본: 박사논문 1434편(어휘 표현은 본문 색인이 있는 1145편) × 교수 298명(KCI 논문 묶음 5967편). 학맥 기준일 2026-11-22. 교수→교수 간선 234건(확정 223 · 확정+추론 231).

## 파일

| 파일 | 행 | 내용 |
|---|---|---|
| theses.csv | 1434 | 제자 박사논문(간선 1건 = 논문 1편): thesis_id(RISS control_no → riss_url), 제목·연도·학위 대학(정규화 표기), 지도교수 id·이름, 통합목록의 지도교수 원표기, 학위자 동일인 판정(확정/추론/미확인 + 원 판정), 어휘 표본 포함 여부, 간선 단계별 계보 뿌리 |
| professors.csv | 298 | 교수: 이름·소속(분석용 정규화·학맥 원표기), 동일인 판정, 본인 박사논문 id, 박사학위 연도(출처: 학맥 간선 / 국중 박사논문 KAC / 이름+학과 유일), KAC·ISNI(A 등급), A 등급 cretId 집합과 판정 규칙, 묶음 논문 수, 표본 안 지도 논문 수, 간선 단계별 지도교수·뿌리 |
| professor_articles.csv | 5967 | 교수 묶음에 들어간 KCI 논문: kci_id(→ kci_url)·cretId·연도·학술지·제목·현대문학 선별 상태. 제목 정규화 중복 제거 뒤 묶음(prelim_similarity.py 규칙) |
| lineage_professor_edges.csv | 234 | 교수→교수 지도 간선(계보 구성의 원자료): 지도교수·제자 교수·논문·연도·대학·동일인 판정 3단계와 원 판정·근거·표본 포함 여부 |
| pair_relations_similarity.csv.gz | 427332 | 논문 × 교수 쌍 전체. same_university, 간선 단계별 관계(relation_all / relation_confirmed / relation_confirmed_inferred ∈ base·advisor·grand·sibling·uncle·lineage_other·same_uni_outside_lineage)와 같은 대학×계보 표지, 표현별 잔차 유사도(resid_<표현>)와 원 유사도(raw_<표현>; lda 는 Bhattacharyya 계수). 표현 = bge, bge_subset, tfidf, lsa300, lda50, ko_sroberta, e5_large, mpnet_multilingual. 어휘 표본(bge_subset·tfidf·lsa300·lda50) 밖 논문은 그 열이 빈칸 |
| thesis_topics_k50.csv | 1145 | 박사논문 본문의 LDA(K=50) 토픽 비중 θ |
| professor_topics_k50.csv | 298 | 교수 묶음의 토픽 비중(논문 θ 평균, 합 1) |
| topics_k50.csv | 50 | 토픽별 상위 낱말 20개와 KCI 전체·현대문학·박사논문 비중 |
| lineage_topics_k50.csv / university_topics_k50.csv | — | 계보 뿌리별·학위 대학별 토픽 비중과 특이 토픽 |
| eta2_k50.json | — | sqrt θ 공간에서 지도교수·계보·대학 묶음의 η² 와 순열 귀무 평균 |
| regression_results.csv | 168 | 표현 5 × 간선 단계 3 의 관계별 계수·SE(논문/교수/양방향)·t·p·Holm p·순열 p·부트스트랩 구간·표준화 효과·쌍 수 |
| decade_results.csv | 360 | 학위 연대별(≤2004 / 2005–2014 / 2015–) 관계 계수 |
| window_results.csv | 420 | 지도교수 묶음의 학위 연도 기준 창별 평균 잔차(전승 방향) |
| trajectory_bins.csv / trajectory_regression.csv | 49 / 28 | 제자 본인의 KCI 논문이 학위 뒤 지도교수·형제·재직 대학 동료 묶음과 얼마나 닮는지(학위 후 연차 구간별) |
| student_articles.csv | 9719 | 제자 궤적의 논문별 자료: kci_id·연도·학위 후 연차·발표 당시 소속·이동 여부·잔차 유사도(초록 없음) |
| methods.json | — | 표본·출처·표현·모형·스크립트·주의 사항 |

## 관계 부호

| 코드 | 뜻 |
|---|---|
| advisor | 지도교수 |
| grand | 조부 교수 |
| sibling | 형제 교수 |
| uncle | 삼촌 교수 |
| lineage_other | 같은 계보 기타 |
| same_uni_outside_lineage | 계보 밖 같은 대학 |
| same_uni_x_lineage | 같은 대학×계보 |
| base | 기준(다른 대학·계보 밖) |

같은 대학×계보: 계보 관계(조부·형제·삼촌·기타)이면서 제자의 학위 대학이 그 교수의 소속이면 1. 회귀에서는 계보 더미와 함께 켜지는 추가 더미다.

## 간선 단계(edge_tier)

교수→교수 지도 간선의 학위자 동일인 판정으로 계보를 제한한 민감도 분석. `all` = 확정+추론+미확인 전체(본 분석), `confirmed` = 확정만, `confirmed_inferred` = 확정+추론. 지도교수→논문 간선은 단계와 무관하므로 지도교수 항은 세 단계에서 같다.

## 재계산 요령

- 주 결과(`regression_results.csv` 의 beta): `pair_relations_similarity.csv.gz` 에서 표현별 `resid_*` 를 종속변수로, `relation_<tier>` 더미 6개 + `same_uni_x_lineage_<tier>` 를 설명변수로 OLS(상수항 없음, 기준 = base). 어휘 표현은 어휘 열이 비어 있지 않은 행만.
- 양방향 클러스터 SE 는 edge_id(논문)와 professor_id(교수) 두 클러스터 분산의 합에서 HC0 를 뺀 것.
- 토픽 η²: `thesis_topics_k50.csv` 의 sqrt θ 를 `theses.csv` 의 지도교수·계보 뿌리·대학으로 묶어 집단 간 제곱합 비율.
- KAC·ISNI 가 빈 교수 중 `kac_shared_with_namesake` 가 있는 행은 동명 인물과 전거를 공유해 판정을 보류한 것이다(교수 묶음은 KAC 가 아니라 cretId 로 만들었으므로 분석에는 영향이 없다).
- 원자료 되찾기: 박사논문은 riss_url, KCI 논문은 kci_url(초록·참고문헌은 KCI 포털·Open API), 인물 전거는 KAC(`https://lod.nl.go.kr/resource/{KAC}`)·ISNI.

## 주의

- 수집 범위(RISS 통합목록 박사논문 + 교수 명단 + 도서관 조회) 안의 관계만 센 값이며 전체 학계가 아니다
- 이름만 같은 저자는 자동 병합하지 않았다. A 등급 cretId·KAC 는 규칙 기반 판정이며 연구자 검토표가 따로 있다
- 학위자 동일인 판정이 미확인인 간선이 전체의 대부분이다(이 표 student_identity_status). 교수→교수 간선의 판정 상태별 재추정(edge_tier)이 regression_results 에 있다
- 조부(3세대) 효과는 표현과 표본 구성에 민감하다(학맥 정정 전 판에서는 확정 간선만으로 bge·tfidf 가 Holm 비유의였고, 정정판에서는 간선 대부분이 확정이어서 단계별 차이가 거의 없다)
- 학맥 정정판: 2026-11-04 교수 본인 박사논문 연결 전수 재검토와 그 뒤 추가 간선을 반영한 계보로 관계 부호·회귀·제자 궤적·계보 토픽을 다시 계산했다. 표본(논문 × 교수)과 유사도는 정정 전과 같다
- 같은-대학 항의 순열 p 는 같은 대학 안에서 섞으므로 정보가 없다(p=1)

이 표의 수치는 수집 자료 안의 연결만 센 값이다. 이름이 같은 저자를 자동으로 합치지 않았으며, 동일인 판정은 3단계로 표시했다.

## 학맥 정정판

이 판은 학맥 기준일 2026-11-22 의 정정된 계보로 관계 부호·회귀·제자 궤적·계보 토픽을 다시 계산한 것이다. 2026-11-22 판은 표본도 새 학맥에 맞춰 다시 잡았다(범위 밖으로 빠진 교수와 그 지도논문을 빼고, 새로 확인된 지도 관계의 논문을 더함; 교수 묶음의 KCI 논문은 그대로). professor_articles.csv 와 professors.csv 의 식별자 열은 이전 판을 옮겼다.