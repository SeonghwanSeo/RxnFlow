# 최종 검토와 QED 실험 — 2026-10-02

## 범위와 현재 상태

사용자 요청: 현재 source 전체를 가독성 중심으로 검토하고 RxnFlow master, CGFlow, HSX explore/main을 참고한다. 역합성 누락을 확인·수정한다. Full catalog에서 QED 및 QED+Lipinski masking을 각각5,000 step 실행하고 optimization 곡선과 개선 항목을 공유한다. QED+Lipinski는 reward 조합이 아니라 MW500/HBA10/HBD5 action masking이며 logP는 제외한다. QED 단독은 기존 MW500/max_atoms50 설정을 유지한다.

비교 reference는 RxnFlow master a39c7ae, CGFlow89fe021, HSX explore250509 2d47244/main e999c3d이다. Config/API, preparation/library, reaction/synthon/property/graph, env/reverse search, policy/categorical/subsampling/replay, models, trainer/sampler/CLI를 검토했다. 첫 성능 감사의 구조를 유지하며 새 추상화 계층을 추가하지 않았다.

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

| 영역 | 개선 후보 | 판단 기준 / 현재 선택 |
| --- | --- | --- |
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

## 검증 기록

- `OMP_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-goal-refactor-quick.log 2>&1`:51 passed/1 deselected,11.26s.
- gnode7 `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 RXNFLOW_ENV_DIR=/home/shwan/Project/HSX/data/envs/enamine_full_20261001 ./test.sh heavy > runs/qed_validation_20261002/heavy.log 2>&1`:1 passed,57.17s.
- 나머지 최종 검사와5k 실험 결과는 완료 후 아래에 추가한다.
