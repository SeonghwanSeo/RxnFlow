# 최종 검토와 QED 실험 — 2026-10-02

후속 수정: [기준 구현 복원 기록](implementation-deviations.md#restoration-implemented--2026-10-02)을 참고한다. 이 문서의 기존 실행 수치와 이전 구현 설명은 당시 snapshot에 해당한다. 후보 압축·관측 action 강제 포함·기존 attention 구조는 이후 제거했으며, 이 수치로 새 구현의 속도나 품질을 판단하지 않는다.


## 범위와 현재 상태

Source 전체 검토, 가독성 정리, 역합성 탐색 누락 수정과 full catalog의 두5,000-step 실험을 완료했다. 두 실험 모두 최종1,024회 평가에서 전부 유효·고유 분자를 생성했고 모든 forward 경로가 재현됐다. QED+Lipinski는 QED reward를 그대로 두고 MW500/HBA10/HBD5 action masking을 적용한 실험이며 logP는 제외한다. QED 단독은 기존 MW500/max_atoms50 설정을 유지했다. 사용자 코드 검토와 추가 production chemistry 수기 검토는 별도로 남는다.

비교 reference는 RxnFlow master a39c7ae, CGFlow89fe021, HSX explore250509 2d47244/main e999c3d이다. Config/API, preparation/library, reaction/synthon/property/graph, env/reverse search, policy/categorical/subsampling/replay, models, trainer/sampler/CLI를 검토했다. 첫 성능 감사의 구조를 유지하며 새 추상화 계층을 추가하지 않았다.

## 5,000-step 결과

학습 전후 각각1,024 raw attempts를 EMA sampling model,temperature1,random exploration0으로 평가했다. 실패를 retry하거나 좋은 분자만 선별하지 않았다. 아래 평균 QED는 유효 분자 기준이다. 학습 곡선에는 random exploration0.05가 포함되므로 최종 평가 수치와 구분한다.

| 항목 | QED | QED + Lipinski masking |
| --- | ---: | ---: |
| 유효 생성 수, 전 → 후 | 638 → 1,024 | 639 → 1,024 |
| 평균 QED, 전 → 후 | 0.5195 → 0.8227 | 0.5252 → 0.7657 |
| 실패를0으로 포함한 평균 QED, 전 → 후 | 0.3237 → 0.8227 | 0.3277 → 0.7657 |
| 최종 고유 분자 수 | 1,024 | 1,024 |
| 최종 MW/HBA/HBD 동시 통과 | 1,021/1,024 (99.71%) | 1,021/1,024 (99.71%) |
| 최종 MW / HBA / HBD 위반 수, 중복 가능 | 3 / 0 / 0 | 3 / 1 / 0 |
| Internal diversity, 전 → 후 | 0.8837 → 0.8801 | 0.8833 → 0.8815 |
| 최종 비어 있지 않은 Murcko scaffold 수 | 1,011 | 1,021 |
| 학습 wall time, 평가 제외·checkpoint 포함 | 3h47m14s | 5h21m13s |
| Update 시간 중앙값, checkpoint I/O 제외 | 2.720s | 3.884s |
| 마지막500 update 평균 시간 | 3.475s | 5.084s |
| Peak allocated VRAM | 908.36MiB | 1,000.40MiB |

Internal diversity는 chirality를 포함한 Morgan2048/radius2 fingerprint의 모든 서로 다른 sample pair에 대해 `1 − mean Tanimoto`로 계산했다. 중복 sample도 포함한다. 이1,024회 평가에서의 높은 고유성은 전체 생성 분포의 무중복을 보장하지 않는다.

| 최종 경로 길이 | QED: 개수 / 평균 QED | Lipinski: 개수 / 평균 QED |
| --- | ---: | ---: |
| 1 reaction | 749 / 0.8538 | 31 / 0.7820 |
| 2 reactions | 268 / 0.7407 | 988 / 0.7658 |
| 3 reactions | 7 / 0.6406 | 5 / 0.6471 |

Lipinski는 약2,500 step 이후 주로 두 BiReaction 경로로 전환됐고 전체 평균 QED가 하락했다가 일부 회복했다. 최종2-reaction 비중은96.5%이며 QED 단독은26.2%다. 최종 전체 QED만으로 masking이나 모델의 우열을 결론 내릴 수 없다. 단일 seed이며 경로 분포·forward action space·backward 근사가 함께 작용한다. 특히3-reaction 표본은 매우 적다. 다음 검토에서는 길이별 reward와 TB residual, 실제 parent-state에 대한 backward 확률을 함께 보는 것이 우선이다.

두 실험은 source commit baa4ca0에서 시작해 설정 변경·재시작 없이 종료했다. Slurm823190은 `COMPLETED, ExitCode=0:0`; QED 프로세스도 최종 평가를 기록하고 종료했다. 모든5,000개 step 로그의 연속성·유한값, model/EMA weights의 유한값, 최종/latest checkpoint의 동일성, config 및 공통 catalog signature, 평가 JSON/JSONL의 일치를 읽어서 검증했다.

로컬 산출물은 `runs/qed_validation_20261002/`에 보존하며 Git에는 포함하지 않는다. 그래프는 `optimization.png`, `optimization.pdf`, `optimization.svg`; 종합 수치는 `final_summary.json`, 검증은 `completion_audit.json`이다. 각 run 폴더의 `training.jsonl`, `evaluation_before/after.jsonl`, `checkpoint_00005000.pt`에 원본 로그·평가 표본·최종 checkpoint가 있다. 모든 곡선의 마지막 step이5,000인지 확인하고 최종 PNG를 직접 검토했다.

## 코드 수정

- Policy의 긴 candidate_batch에서 block encoding·matrix scoring만 `_score_blocks`로 분리했다. 상태/후보 준비, replay index 정리, neural scoring의 세 단계에 주석을 붙였다. 수학적 동작과 RNG 순서는 유지했다.
- Reverse 결과 두 개 제한을 제거했다. RDKit 결과는 이미 전부 계산·중복 제거한 뒤 잘라 버리고 있었고, 앞선 두 분해가 catalog에 없으면 뒤의 유효 분해를 놓쳤다.
- 전역 최소 깊이 가지치기를 제거했다. 먼저 찾은 짧은 경로가 다른 긴 경로를 지우고 template 순서에 따라 결과가 달라질 수 있었다. 현재 reaction-count 상한은 그대로 유지한다.
- Known branch 보존, canonical dedup, catalog lookup, forward 재실행 확인과 깊이 가중 backward 확률은 유지했다. Template 밖의 화학이나 property-budget까지 정확하게 반영한 역 MDP를 증명하는 구현은 아니다.
- Raw rollout의 batch uniqueness와 평균 반응 수를 training.jsonl에 기록한다. 실험용 QED reward는 최종 분자의 QED·MW/HBA/HBD·통과율을 추가 기록하며 reward는 변경하지 않는다.

## 역합성 근거

실제 RDKit 합성 테스트에서 hexane의 다섯 분해 중 세 번째만 catalog-supported인 경우를 재현했다. 이전 두 후보 제한은 이 경로를 누락한다. 다른 테스트는 terminal 생성물이 direct brick 경로와 보호된 brick의 activation 경로를 모두 가질 때, 짧은 경로 때문에 긴 경로가 지워지지 않는지 검증한다.

현재 min-size10 catalog와 과거 replay의118개 유효 생성물로 비교했다. 이전180개 경로 → 수정291개 경로,65개 생성물에서 추가 경로. Serial 탐색 시간 합은1.3506초 →1.5371초. 이는 해당 표본의 결과이며 모든 분자에 대한 exhaustive ground truth 또는 장기 runtime 보장은 아니다. 과거 policy/action index는 재사용하지 않았고 분자 구조만 감사 입력으로 사용했다. 원시 데이터는 로컬 runs/qed_validation_20261002/retro_audit.json에 있다.

## 실험 설계

- 동일한 full min-size10 환경(895개 library),seed0,64 fresh+64 replay,ratio0.01/min10,hidden128/3layers,최대3 reactions,TB exponent32,EMA0.99.
- QED: gnode7 physical GPU0,기존 allocation CPU0–3. QED+Lipinski: 별도 Slurm GPU1개/CPU8개 job,노드 gnode7,Slurm이 지정한 GPU 사용.
- 각각 처음부터5,000 update.500 step마다 checkpoint. 학습 전후 각각1,024 raw rollout을 별도 RNG로 평가해 학습 RNG를 보존한다. 실패를 retry로 숨기지 않고 모든 attempt를 기록한다. 유효 경로는 forward로 재실행해 terminal SMILES를 확인한다.
- 실행 코드는 commit snapshot으로 고정한다. JSONL은 매 step,stdout은50 step마다 기록한다. 곡선에는 QED,validity,uniqueness,최종 property 통과율,TB loss,반응 수,runtime을 포함한다.

## 개선 후보와 우선순위

다음 구현 우선순위는 **반응별 property budget 보정 → backward/경로 길이별 학습 검토 → 후보 index 처리의 tensor 연산 확대 → 입체화학 feature 보완**이다. Replay DB와 artifact 저장 방식 변경은 현재 규모에서 후순위다. 아래 항목은 후속 제안이며 이번 실험 도중 의미나 구현을 변경하지 않았다.

| 영역 | 개선 후보 | 판단 기준 / 현재 선택 |
| --- | --- | --- |
| Property budget | Template의 생성·소모 원자와 결합 변화에 따른 보정 | Synthon 합만으로는 반응이 추가하는 fragment를 빠뜨린다. 아래 실측 예제처럼 MW/HBA/HBD와 capacity 예측 모두에 영향을 준다. Synple 상수를 복사하지 않고 현 template에 맞는 보정부터 검토한다. |
| Backward MDP | reaction_count를 포함한 정확한 parent-state와 budget admissibility | 현재 깊이 가중은 근사다. 이번 수정은 탐색 누락을 줄였지만 정확한 backward MDP와 동일하다는 주장은 하지 않는다. 정확도 요구와 비용을 먼저 결정해야 한다. |
| Replay | Mol/SMILES·중간체의 중복 저장을 줄인 columnar 배열 또는 SQLite trajectory store | 현재10k FIFO는 단순하고 충분히 작다. 장기 실험의 checkpoint 크기/저장 시간과 중복률을 측정한 뒤 도입한다. DB 자체는 학습 속도 개선을 보장하지 않는다. |
| Env artifact | 압축 NPZ 대신 mmap 가능한 NPY 배열과 별도 ID/provenance 저장 | 현재 로딩 약11초. 더 큰 catalog·여러 worker/experiment의 중복 RAM이 문제가 될 때 가치가 있다. Schema를 바꾸기 전에 실측한다. |
| Policy | 동일 attachment/type eligibility의 scoring group 묶기, block feature transfer 재사용 | 작은 library별 matmul/kernel/Python 비용이 남는다. Dense 전체 조합은 낭비가 커서 grouping의 복잡성과 이득을 비교해야 한다. |
| Masking | Dense state×sampled-block 일괄 계산 | 이미 비교에서93.9→93.3ms로 이득이 없고8.4배 조합을 계산했다. 현행 유지. |
| Model features | atom chirality와 bond E/Z를 명시적으로 구분 | 현재 표현은 일부 stereoisomer와 fingerprint collision을 구분하지 못한다. Feature 범위 변경이므로 별도 실험으로 검증한다. |
| Block model | Synthon graph/anchor representation 또는 graph→fingerprint pretraining | 위치/구조 정보 보존의 개선 후보지만 추가 준비·학습 비용이 든다. 현 uint8 fingerprint baseline을 먼저 평가한다. |
| Chemistry | UniReaction 확장·template substrate 범위 수기 검토 | 자동 forward/reverse roundtrip은 실제 합성 가능성을 보장하지 않는다. 사용자 후속 화학 검토가 필요하다. |
| Exploration | 경로 길이별 coverage, invalid 원인, 반복 분자와 reward 편중 | QED만으로 단순·짧은 경로에 집중할 수 있다. 이번 곡선/평가에서 확인 후 subsampling 또는 objective 변경을 결정한다. |
| Evaluation | 다중 seed·다른 reward·holdout BB/task | 이번 두 단일-seed5k 실험은 작동/최적화 검증이다. 모델 우열이나 일반화 결론을 내리려면 추가 실험이 필요하다. |
| Learning diagnostics | 경로 길이별 QED·TB residual과 valid/invalid loss 분리 | 전체 평균만으로는 action-space 구성 변화와 within-length 학습 품질을 분리하기 어렵다. 길이 조건부 평가와 backward 근사의 영향을 먼저 확인하고 objective 변경 여부를 결정한다. |

## 검증 기록

- `OMP_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-goal-refactor-quick.log 2>&1`:51 passed/1 deselected,11.26s.
- gnode7 `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 RXNFLOW_ENV_DIR=/home/shwan/Project/HSX/data/envs/enamine_full_20261001 ./test.sh heavy > runs/qed_validation_20261002/heavy.log 2>&1`:1 passed,57.17s.
- `OMP_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-goal-handoff-quick.log 2>&1`:51 passed/1 deselected,14.36s;compile/lint/build/import 통과.
- `PYTHONPATH=src .venv/bin/python runs/qed_validation_20261002/verify_final.py > runs/qed_validation_20261002/completion_audit.log 2>&1`:exit0;두 run의 완료 artifact 일치 검증.
- `PYTHONPATH=src .venv/bin/python codex/scripts/summarize_validation.py runs/qed_validation_20261002/qed runs/qed_validation_20261002/qed_lipinski --output runs/qed_validation_20261002/final_summary.json`:exit0;구조 다양성·길이별 QED·최종 replay 분포 집계.
- `MPLCONFIGDIR=runs/qed_validation_20261002/matplotlib_cache .venv/bin/python codex/scripts/plot_optimization.py runs/qed_validation_20261002/qed runs/qed_validation_20261002/qed_lipinski --output runs/qed_validation_20261002/optimization`:최종5k PNG/PDF/SVG/JSON 생성 완료.
- `git diff --exit-code baa4ca0 -- src`:exit0;실험 snapshot과 현재 production source가 동일함을 확인.

## 실행 초기 확인

실행 commit은 baa4ca0, QED tmux는 rxnflow-qed-5000/physicalGPU0, Slurm job은823190/physicalGPU1이다. Slurm에서 CPU4–11을 배정받았고 QED는 기존 allocation CPU0–3을 사용한다. 실행 전1,024 raw attempts에서 QED는638개, QED+Lipinski는639개 유효 생성물이 있었으며 모든 유효 경로의 forward 재실행이 일치했다. 최적화 완료 결과와 구분한다.

최종 분자의 strict MW/HBA/HBD 통과율은 초기 각각77.9%/79.8%였다. QED+Lipinski의639개 유효 분자 중 MW500 초과127개,HBA10 초과9개,HBD5 초과1개가 있었다(중복 가능). Action budget은 synthon property의 합이며 실제 product descriptor의 hard filter가 아니다. 결합/UniReaction에서 추가되는 원자·작용기의 property 변화가 반영되지 않을 수 있다. 따라서 reaction별 검증된 budget 보정 또는 별도 final eligibility의 필요성을 검토할 항목으로 추가한다. 이번 실험의 합의된 masking 의미는 바꾸지 않는다.

현재 template를 작은 입력에 실행해 측정한 `생성물 − 입력 synthon 합`은 아래와 같다. 실제 input/output 및 전체 descriptor는 runs/qed_validation_20261002/budget_examples.json에 있다. HBA/HBD는 결합 환경에 따라 달라질 수 있으므로 이 예제의 수치를 모든 기질에 적용하는 상수로 하드코딩하면 안 된다. 현재 state descriptor는 매번 재계산하므로 과거 반응에서 추가된 부분은 다음 단계 예산에 반영되지만, 선택하려는 반응에서 새로 추가되는 부분은 빠진다. Heavy-atom 초과는 선택 후 structural check로 거부되며 MW/HBA/HBD는 최종 hard check가 없다.

| Template / 예제 | ΔMW | ΔHBA | ΔHBD | Δheavy atoms |
| --- | ---: | ---: | ---: | ---: |
| rxn1: amide, `CN[1*]` + `*CC` | 28.010 | 0 | 0 | 2 |
| rxn4: sulfonamide, 같은 입력 | 64.065 | 1 | 0 | 3 |
| rxn21: `CC[14*]` + `*CC` | 133.114 | 4 | 1 | 10 |
| nitrile_to_tetrazole: `CC[11*]` | 69.047 | 3 | 1 | 5 |

별도32회 강제3-reaction rollout에서는4개 유효 생성물이 나왔고, 역합성 경로는11→13개(1개 생성물에서 증가)였다. 이는 짧은 경로에 편중된 기존 replay 표본을 보완하지만 표본이 작아 포괄적인 화학 coverage 결론은 내리지 않는다. 파일은 runs/qed_validation_20261002/retro_three_step_audit.json이다.

위118개 생성물 중 이전 탐색에서 경로가0개인 경우는42개였고 수정 후에는0개였다. 어떤 표본도 경로 수가 줄지 않았다. 강제3-reaction 표본에서도 이전1개의 unreachable 결과가 수정 후0개가 되었다. 생성 경로를 known branch로 주입하지 않은 독립 탐색 비교이므로, known branch 보존이 가렸던 누락을 드러낸다.

## Replay 저장 형식에 대한 실측

QED step500 checkpoint는24,829,248bytes(약23.7MiB)이며 replay10,000개 trajectory를 담는다. State occurrence20,463개 중 unique SMILES는9,337개다. 반복되는 초기 empty state를 제외하면 분자 state의 중복은 제한적이다. 유효 terminal9,969개 중 unique는9,965개였다. 현재 규모에서는 SQLite/별도 DB 전환보다 chemistry budget, backward 근사, stereochemistry 표현을 우선 검토하는 편이 낫다. Replay를 훨씬 크게 유지하거나 cross-run 검색·재활용이 필요해질 때 columnar/DB 저장의 가치가 커진다. 근거는 로컬 runs/qed_validation_20261002/replay_audit_500.json이다.

## 학습 중 runtime 관찰

Step201–300의 평균 update 시간은 QED1.64초/Lipinski1.75초였고, 각 실험이 약1,200/1,000 step인 시점의 최근100 step은2.09초/2.63초였다. Rollout과 그 이후 학습 구간 모두 증가했다. 그 시점 gnode7 available RAM은 약83GiB, 주 프로세스 RSS는 각각약3.8GiB여서 메모리 부족으로 판단할 근거는 없다. GPU 순간 사용률은5%/9%였지만 한 번의 측정이므로 평균 utilization으로 해석하지 않는다.

Checkpoint500→1000의 replay action 분포도 바뀌었다. Lipinski의 첫 brick type21은85.6%→31.8%로 줄고 type8,3,1 등으로 분산됐다. QED는 type8이 약76%로 유지됐다. 반응 횟수가 비슷해도 state의 synthon type에 따라 score해야 하는 library와 후보 수가 달라질 수 있다. Action mix 변화가 runtime 차이의 가능한 설명이며 아직 인과관계를 profiling으로 분리한 결과는 아니다. 다음 성능 검토에서는 평균 step 시간만 비교하지 말고 sampled library 수·state/library group 수·후보 pair 수·reverse 시간도 함께 기록하는 것이 좋다. 근거는 runs/qed_validation_20261002/action_mix_audit.json이다.

추가로 저장된 replay에서 동일 seed로128개씩 골라 CPU candidate 준비/scoring을 profiling했다. Lipinski checkpoint500→2000에서 transition 수는262→263으로 거의 같지만 available state/reaction/library group은8,599→22,732개, mask 이후 candidate pair는1,066,984→1,617,858개로 늘었다. `_score_blocks` 누적 시간은0.52→0.88초, candidate_batch 자체 Python 실행은0.10→0.34초, graph encoder는0.63→0.65초였다. 따라서 이 표본에서는 후보 처리 부하 증가를 확인했다. CPU/no-grad 단회 profile이므로 GPU 전체 update의 속도 비교로 사용하지 않는다. QED 첫 profile에는 초기화 비용도 들어가므로 그 total time의 전후 비교는 생략한다. 실행 중인 학습은 변경하지 않았으며 스크립트·통계·cProfile 결과는 runs/qed_validation_20261002/profile_candidates.py, candidate_profile.json/log 및 candidate_*.prof에 있다. Library 내부의 query/mask index 구성까지 tensor로 묶는 것이 후속 성능 개선 후보이며, dense 전체 catalog 계산을 다시 도입할 이유는 없다.

이후에는 경로 길이 변화도 나타났다. Lipinski step2001–2500의 평균 QED0.819/반응 수1.137에서 step3001–3242의0.736/1.755로 바뀌었고, 평균 update 시간은3.59→4.76초였다. 같은 구간의 QED 단독 실험은 주로1-reaction 경로와 평균 QED약0.82를 유지했다. 초기 상승만으로 안정적인 수렴을 선언할 수 없으며, 최종 평가는 길이별 QED와 reaction 사용 분포를 함께 기록한다. 한 seed의 두 run만으로 masking 자체나 backward 근사가 하락의 원인이라고 단정하지 않는다.

Lipinski checkpoint3500의 유효 replay9,848개 중7,618개가2-reaction 경로였다. 가장 흔한 경로들은 rxn3→rxn3, rxn3→rxn1처럼 linker를 거치는 두 BiReaction이었다. 단순 탈보호 횟수 증가만으로 생긴 현상은 아니다. 그 replay의 길이별 QED는1회0.794/2회0.713/3회0.470이고, QED checkpoint4000은1회0.844/2회0.657/3회0.571이었다. 시점과 표본 분포가 달라 실험 우열 판단에는 사용하지 않으며, 평균 reward와 경로별 품질을 구분해야 한다는 근거다. 원시 집계는 runs/qed_validation_20261002/late_path_audit.json이다.

## 2026-10-02 — Actual CUDA update profiling

User requested an actual GPU profile to separate host overhead from GPU computation. Ran QED and QED+Lipinski checkpoints500/5000 serially on gnode7 physical GPU0,CPU0–3,Torch/BLAS1thread,four reverse workers,using the frozen baa4ca0 experiment source. Each case restores model/EMA/optimizer/scheduler/replay/RNG,then executes three warmup updates,five ordinary timed updates,and one separate CPU/CUDA profiler update. These are complete64fresh+64replay updates including chemistry,reverse analysis,reward,TB forward/backward,optimizer and EMA. Only diagnostic in-memory models change;original run artifacts are untouched. Environment load and profiler export/analysis are excluded from update timing.

| Checkpoint | Ordinary update median (5 samples) | Profiled update wall time | GPU active time within that trace | Active fraction of traced wall time |
| --- | ---: | ---: | ---: | ---: |
| QED500 |1.577s|2.483s|130.4ms|5.3%|
| QED5000 |3.525s|3.866s|165.0ms|4.3%|
| QED+Lipinski500 |1.703s|2.765s|146.4ms|5.3%|
| QED+Lipinski5000 |4.887s|5.650s|175.5ms|3.1%|

GPU active time is the union of CUDA kernel/memcpy/memset intervals within the CPU UPDATE range,not a sum of nested operator times or an SM occupancy estimate. Profiler overhead inflates wall time,so these fractions describe traced updates only;do not report the remaining95–97% as pure Python time or as an exact fraction of the historical training runs. Nevertheless,the small GPU active durations relative to both traced and ordinary updates establish that these workloads are host-bound rather than dominated by GPU compute. Host work includes Python candidate/group/index construction,CPU tensor operations and dispatch,CUDA launches,chemistry/reward and reverse-worker waits.

For QED500→5000,the profiled train.candidates CPU wall time grows0.952→1.720s,while associated CUDA operator time grows32.2→39.7ms. Across the update,aten::index calls increase26,502→71,396 and aten::arange calls41,054→88,046. The corresponding train.candidates self CPU time grows356→594ms and train.block_scoring self CPU136→293ms. These self times exclude nested recorded operations but still include uninstrumented host work;they are not a pure Python interpreter measurement. This supports prioritizing tensorized/grouped candidate indexing and fewer small operations/kernel launches over GPU model compute optimization. Retrosynthesis wait also contributes:0.215→0.336s in these QED traces.

A bounded1Hz nvidia-smi record is saved as gpu_profile/hardware.csv;thermal slowdown flags were inactive in observed samples. Idle trace-export/analysis periods can have0MHz and clock reason0x1(idle),which is not thermal throttling. This short diagnostic cannot establish thermal history during the earlier full training runs.

Reproduction script,CPU/CUDA traces,operator tables,ordinary timing samples,hardware log and interpretation notes are under runs/qed_validation_20261002/gpu_profile/. See README.md for the exact gnode7 command. summarize_traces.py uses CPU user_annotation ranges only;GPU annotation ranges repeat names and must not be added to CPU phase durations. The final results.json is the authoritative corrected phase summary.

## 2026-10-02 — Synchronous CPU section breakdown

User requested CPU-only section timing to remove asynchronous attribution. Ran the same frozen source/full catalog/checkpoints on gnode7 with CUDA disabled,Torch/BLAS1thread and reverse workers0. Each case restores checkpoint model/EMA/optimizer/scheduler/replay/RNG,warms up3updates,and measures5complete64fresh+64replay updates with lightweight perf_counter scopes. Two policy methods are instrumented only in memory,without changing statement order or expressions. Candidate logits/importance/required positions match the original candidate method on replay states. Source and original artifacts remain unchanged.

The table reports mean seconds/update aggregated across rollout and training. Rows are exclusive and partition the full update;parent times are not added again. Neural-network work now executes on CPU,and reverse search is serial,so these totals must not be treated as a decomposition of historical CUDA times.

| Section | QED500 | QED5000 | Lipinski500 | Lipinski5000 |
| --- | ---: | ---: | ---: | ---: |
| Library sampling | 0.166 | 0.280 | 0.196 | 0.293 |
| Property budget comparison | 0.082 | 0.143 | 0.148 | 0.233 |
| Mask to valid indices/weights | 0.166 | 0.755 | 0.157 | 0.892 |
| Action/query/replay index packing | 0.037 | 0.210 | 0.132 | 0.270 |
| Block features and scoring indices | 0.317 | 0.568 | 0.228 | 1.043 |
| State properties/graph features/groups/batching | 0.156 | 0.209 | 0.153 | 0.332 |
| Graph/block/query neural forward | 0.551 | 0.731 | 0.563 | 0.861 |
| Score matrix products and gather | 0.052 | 0.087 | 0.056 | 0.099 |
| Autograd backward | 1.309 | 2.751 | 1.615 | 2.503 |
| Serial retrosynthesis | 0.615 | 1.359 | 0.444 | 2.263 |
| Forward chemistry and reward | 0.135 | 0.150 | 0.129 | 0.185 |
| Optimizer/EMA and replay | 0.012 | 0.013 | 0.012 | 0.014 |
| Other control, log-probability/TB loss and outputs | 0.090 | 0.216 | 0.082 | 0.307 |
| Total | 3.689 | 7.472 | 3.914 | 9.296 |

The host-side increase is concentrated in mask-result/index construction and block feature/scoring-index packing,not property comparisons alone. For Lipinski,mask-to-index work grows0.157→0.892s,block feature/scoring-index work0.228→1.043s,and action/query/replay index packing0.132→0.270s. These three parts total0.516→2.205s. Property comparison grows0.148→0.233s,and library sampling0.196→0.293s. Pure score matrix/gather work remains0.056→0.099s even on CPU. The subsequent reference audit supersedes the proposed compression optimization: remove valid-index packing and restore masked protocol matrices (see implementation-deviations.md).

Serial reverse search also grows0.444→2.264s for Lipinski. It is a separate CPU-compute contributor;the actual CUDA training uses four workers,so this is not its observed worker-wait time. CPU backward takes1.615→2.503s and includes CPU neural/autograd computation;the earlier CUDA trace is the appropriate evidence for GPU backward compute. No optimization was applied during this measurement.

All individual updates,full inclusive/exclusive section paths and means are saved in runs/qed_validation_20261002/cpu_sections/results.json,summary.json and sections.md. README.md documents section definitions and the exact reproduction command. Every measured update asserts that exclusive section times sum to update time. Validation:GPU-disabled profiling and summary both exit0;OMP_NUM_THREADS=1 ./test.sh quick exits0 with51passed/1deselected in11.20s.

## Reference-fidelity correction

The later source audit found that the earlier “reference-aligned” description was incomplete. Candidate compression was introduced locally,and other differences affect normalizer estimation,exploration,sampling placement,reverse/model overlap and graph architecture. See [implementation-deviations.md](implementation-deviations.md) for confirmed source comparisons,agreed exceptions and correction order. Successful5k runs and passing tests establish execution,not equivalence to RxnFlow master/CGFlow/HSX. Existing profiling results remain measurements of the current implementation,not those reference implementations.
