# HSX 선택적 이식과 검토 기록

2026-10-02 기준 구현 복원 내용과 검증은 [변경점 검토](implementation-deviations.md#restoration-implemented--2026-10-02)에 기록했다. 아래 과거 실험 수치는 당시 source snapshot의 결과다.

`2aac819`에서 시작한 [섹션별 비교표](reference-comparison.md)에 수식, feature, 기본값의 일치 여부를 별도로 기록했다. 이후 사용자 요청으로 bond stereo, temperature 초기값, mask margin을 main에 맞췄다. Property scale은 사용자가 단순한 값으로 조정한 설정이며 유지한다. 아래의 “참고 구현”은 해당 부분의 출처를 뜻하며 HSX main과 모델 전체가 동일하다는 뜻은 아니다.

## 기준과 범위

현재 목표는 hsx의 공개 가능한 모델·학습 로직을 현재 선형 Enamine synthon 환경에 이식하는 것이다. `source/rxnflow_hits`의 작업 트리를 변경하지 않고, `explore_250509` (`2d472443806ac107b9cdc8a65d03866302394d84`)와 로컬 `main` (`e999c3d1b2f911d1b33fb8245c0a66a2946f14b0`)을 비교했다. Backward는 공개 RxnFlow 로컬 `origin/master` (`a39c7ae`)를 기준으로 한다. 전체 production env, 추가 수기 template, workflow library, tier/clustering은 포함하지 않는다.

## 선택한 구현

| 부분 | 참고 구현 | 현재 선택과 차이 |
| --- | --- | --- |
| Block encoder | main의 `models/layers.py:BlockEmbedding` projection + explore fusion MLP 순서 | FP와 property 각각 Linear/LayerNorm으로 projection하고 type embedding과 concat한 뒤 MLP. Projection은 main의 Xavier/zero-bias 초기화. Tier 제외, block_dim128. Fusion hidden layer는 기존 한 층 유지. |
| Model 크기·초기화 | main의 graph/block 크기, MLP output/embedding 초기화 | Hidden128/heads2/layers4, block128. MLP hidden은 Kaiming, output은 Xavier/zero bias. Reaction/type embedding은 uniform[-0.1,0.1]. Fusion/policy는 Linear→LN→SiLU 유지, GENConv bias와 empty state는 사용자 요청으로 보류. |
| Reaction conditioning | explore의 `hook_firstblock`, `hook_birxn` | State + reaction embedding에 SiLU를 적용하고 FirstBlock/BiReaction별 MLP. Workflow/order 대신 reaction name으로 식별. Graph를 reaction마다 다시 계산하지 않음. |
| Action similarity | main의 `models/layers.py:SimilarityMDP(dot)` | Block embedding만 L2 정규화하고 query와 dot product. Reaction별 bounded temperature 0.01–10, 초기 0.2. 별도 클래스/선택 옵션 없이 현재 모델에 직접 구현. |
| UniReaction | 현재 합의한 동적 MDP | State + reaction embedding의 scalar head. 동일한 bounded temperature convention을 적용하고 BiReaction/block과 하나의 categorical에서 경쟁. 두 hsx 버전의 workflow-determined placeholder와 다름. |
| Graph readout | explore의 molecular mean + virtual node | 2H concat → LayerNorm. 추가 2H→H compression 제거. |
| Attention 구현 | RxnFlow master/CGFlow/HSX explore | GENConv(add), TransformerConv, graph-mode normalization, conditional scale/shift를 native Torch로 구현. Fixed padding과 virtual node 유지. 출력·gradient를 독립 수식으로 비교. |
| Mask | explore의 sampled-row 적용 + main의 1% margin | 가능한 type의 library를 먼저 uniform subsampling한 뒤, 선택된 row에 적용. Positive bound는 `< limit × 1.01`; negative bound는 `< limit + abs(limit) × 0.01`; zero bound는 `<= 0`. Heavy-atom capacity는 항상 strict. |
| Bond stereo | main의 `utils/vocab.py:BondFeaturizer` | NONE/ANY/Z/E/CIS/TRANS/unknown categorical로 인코딩. 기존 bond type·conjugation·ring과 합쳐 13차원. |
| Synple property 보정 | explore의 `envs/building_block.py` | At isotope 차감 및 linker MW +29는 사용하지 않음. Dummy label 질량은 제외하고 state/block 모두 Descriptors.ExactMolWt 사용. 기존 MolWt 기반 feature NPZ는 재생성 필요. |
| TB loss/reward | main의 `gflownet/algo/trajectory_balance.py` | `mean((logZ + ΣlogPF - ΣlogPB - exponent × log(max(raw_reward, floor)))²)`. Invalid raw reward는 0. Local injectable reward 유지. |
| Optimizer | main의 `gflownet/online_trainer.py` | AdamW 두 parameter group. Policy와 logZ learning rate 분리, 공통 `2^(-step/lr_decay_steps)` decay. Policy gradient만 global norm 100으로 clip. 사용자 선택으로 random probability 0.1, reward floor 1e-4. Weight decay는 기존 1e-8 유지. |
| Replay | main의 FIFO buffer와 data source 순서 | 균일 비복원 추출. 기존 buffer에서 추출한 다음 fresh trajectory 추가. Warmup service/data-source abstraction은 도입하지 않음. |
| Sampling model | explore의 EMA | `target = decay × target + (1-decay) × model`. Checkpoint에 model과 EMA 모두 보관. |
| Subsampling probability | RxnFlow master의 별도 numerator scoring | 관측 action과 독립인 library draw로 분모 추정. 관측 action은 별도 scoring하며 logP≤0 clamp. Mask된 column은 -inf로 유지. |
| Backward | public RxnFlow | Known branch 보존, template reverse search, depth-weighted probability. HSX의 `logPB=0`은 이식하지 않음. |

## Action과 실패 처리

Policy candidate에는 reaction과 oriented block index가 있고 product SMILES는 비어 있다. Candidate scoring은 chemistry를 실행하지 않는다. 선택한 action만 `env.step`에서 실행한다. 성공하면 기존 state를 변형하지 않고 새 Mol을 반환하고, trajectory action에 canonical product SMILES를 기록한다. 선택된 반응의 site signature 또는 실제 `max_atoms` 조건이 실패하면 `InvalidTransition`으로 처리한다.

실패 action도 trajectory에 남고, 빈 product SMILES 및 `logPB=0`인 failure sink로 학습한다. Prefix만 남기면 실패 action의 forward 확률이 TB loss에서 빠지므로, 이식 과정에서 이 부분을 수정했다. 후보가 전혀 없는 state에는 선택할 action 자체가 없으므로 현재 prefix를 invalid로 기록한다. 예상하지 못한 template ambiguity/model 오류는 invalid chemistry로 덮지 않는다. 재선택이나 mask 해제 fallback은 없다.

Property budget 합은 정확한 terminal property 검사가 아니다. 특히 synthon에서 생략한 원자가 결합 때 삽입될 수 있고 logP 등은 비가산적이다. 선택 후에는 구조와 graph capacity만 강제하며, 평가에서는 정확한 terminal bound 초과 수를 별도로 기록한다. UniReaction에 추가 block budget은 없다.

## Backward와 확률의 범위

성공한 전이마다 생성 경로를 known branch로 먼저 넣는다. Reverse SMARTS에서 rule별 중복 제거된 canonical precursor set을 모두 추가하며, BiReaction의 block은 실제 oriented catalog row에 있어야 한다. 후보를 forward로 다시 실행해 동일 생성물을 만드는지 확인한다. 시작 brick까지 연결되는 경로를 반응 횟수 상한 내에서 재귀적으로 찾는다. 후속 감사에서 두 후보 제한과 최소 깊이 가지치기가 경로를 누락함을 확인해 제거했다.

Parent branch의 weight는 그 parent에서 시작점까지의 leaf 경로에 대해 `Σ C^(-depth)`이며, `C`는 전체 catalog/reaction에서 계산한 기준 action 수다. 이 weight들을 정규화해 `PB`를 얻고 `(action, parent_smiles)`로 실제 branch를 찾는다. Search가 모든 경로를 열거하거나 forward budget mask 및 정확히 동일한 reaction count까지 검증하지는 않는다. 이것은 사용자가 허용한 근사 범위이며, 정확한 현재 MDP의 backward distribution이라는 주장은 하지 않는다. 실패한 전이에는 reverse chemistry를 수행하지 않는다.

## 재현과 검증

자동 검증은 synthetic fixture만 사용한다. Scoring의 block norm 불변성, mixed Uni/Bi logit의 gradient, MSE와 backward 부호, 실패 action 학습, mask tolerance와 0/음수 상한, bounded reverse search/known branch, scheduler·dropout RNG·EMA를 포함한 정확한 재시작을 확인한다. Heavy 검사는 기존 1만 개 환경에서 train → restart → sample 경로를 확인한다.

변경 전 비교 코드는 로컬 ignored `runs/hsx_port_baseline_source/`에 SHA256 inventory와 함께 보존했다. 이 baseline도 이미 budget-mask-before-subsampling을 사용한다. 예전의 candidate-product 검사 pilot과 혼동하지 않는다. 변경 후 비교는 동일한 source env, seed, 30 updates, 64 raw evaluation attempts와 sampling 설정을 사용한다. 모델·loss·optimizer·invalid/replay 처리가 함께 바뀌므로 결과를 한 요소의 인과적 효과로 해석하지 않는다. 장기 수렴이나 다중 seed 성능을 증명하는 평가도 아니다.

## 1만 개 환경의 비교 결과

두 실행은 gnode7 CPU, 동일한 `enamine_random_10000_seed0` 환경에서 training seed 0, evaluation seed 11, 30 updates, 평가당 64 raw attempts를 사용했다. 공통 설정은 hidden 64, 4 heads, 2 layers, QED exponent 8, max atoms 50, reaction 1–3회, MW budget 500, subsampling ratio 0.002/min 10, policy learning rate 0.001, EMA 0.9, batch 4 + replay 4다. Revised에는 logZ learning rate 0.1과 half-life 20000이 추가됐다. 같은 seed라도 architecture와 RNG 소비가 달라 분자별 paired experiment는 아니다.

| 지표 | Baseline 학습 전 | Baseline 30 updates 후 | Revised 학습 전 | Revised 30 updates 후 |
| --- | ---: | ---: | ---: | ---: |
| 유효 생성 / 64 | 44 | 62 | 36 | 57 |
| 유효 분자 내 unique | 44 | 62 | 36 | 57 |
| 유효 분자 평균 QED | 0.4857 | 0.6088 | 0.4353 | 0.5973 |
| 전체 시도 평균 QED (invalid=0) | 0.3339 | 0.5898 | 0.2448 | 0.5320 |
| 평균 pairwise Tanimoto distance | 0.8715 | 0.8848 | 0.8787 | 0.8777 |
| 유효 경로 1/2/3 reaction 수 | 22/9/13 | 42/15/5 | 19/9/8 | 44/9/4 |
| 실제 terminal MW > 500 | 11 | 4 | 9 | 4 |

Training elapsed는 baseline 66.15초, revised 68.09초다. Main-process peak RSS는 각각 615.43 MiB와 676.03 MiB이며 reverse worker 메모리 합계는 아니다. 두 실행 모두 정확히 30번의 finite-loss update와 raw trajectory 결과를 저장했다. Revised 학습 후 실패 7건은 전부 전체 budget-feasible 후보가 없는 state였다. 학습 전후에 invalid 시도를 재시도하거나 결과를 property로 사후 필터링하지 않았다.

이 짧은 실행에서는 revised가 baseline보다 우수하지 않았다. Revised 내부의 유효율/QED는 학습 전보다 상승했지만, 변경 묶음의 장기 성능 우위나 개별 변경의 효과를 증명하지 않는다. 이 결과를 숨기거나 pilot에 맞춰 추가 튜닝하지 않고 사용자 검토 자료로 남긴다.

Terminal MW 상한 초과는 양쪽 모델에 남는다(revised 학습 후 최대 536.628, baseline 526.416). 이는 synthon property 합을 사용하는 현재 합의와 일치하는 추정 한계이며, model 검증에서 실제 생성물의 물성 상한을 보장했다고 주장하지 않는다.

### Uni/Bi scoring 진단

고정 state `[11*]CC`에서는 하나의 tetrazole UniReaction이 287개의 sampled BiReaction/block 후보 및 그 inclusion correction과 경쟁했다. Revised에서 unary raw logit은 0.0736 → 0.2804, BiReaction raw logits 범위는 [-0.7322, 0.3330] → [-3.1683, 3.7090]이었다. Unary 선택 확률은 0.000321 → 0.000037로 낮아졌다. 이는 finite logit/gradient 연결은 동작하지만, 단일 action이 많은 block과 경쟁하는 현재 flat action distribution에서 unary exploration을 보장하지 않음을 보여준다. Reaction-first hierarchical sampling이나 group-size normalization을 임의로 추가하지 않았다.

보호된 state `[33*]NCC`의 유일한 Boc deprotection action 확률은 1이었다. Revised 학습 후 raw rollout에는 UniReaction 5회(methyl hydrolysis 3, Boc 1, ethyl hydrolysis 1)가 포함됐다. 학습된 reaction temperature 범위는 0.9816–1.0155로 경계 포화가 없었다. Synthetic test에서는 Uni/Bi가 동시에 있을 때 두 head의 temperature로 gradient가 전달되는 것도 확인했다.

### 재현 명령과 증거

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 gnode7 'cd /home/shwan/Project/HSX && PYTHONPATH=runs/hsx_port_baseline_source/src OMP_NUM_THREADS=1 .venv/bin/python runs/hsx_port_baseline_source/evaluate_subset.py --env-dir data/envs/enamine_random_10000_seed0 --output-dir runs/hsx_port_baseline --steps 30 --count 64'
ssh -o BatchMode=yes -o ConnectTimeout=10 gnode7 'cd /home/shwan/Project/HSX && PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python codex/scripts/evaluate_subset.py --env-dir data/envs/enamine_random_10000_seed0 --output-dir runs/hsx_port_revised --steps 30 --count 64'
```

기존 결과를 덮지 않으므로 재실행할 때는 새 output-dir을 사용한다. Local ignored 산출물은 `runs/hsx_port_baseline/`, `runs/hsx_port_revised/`, 두 report를 교차 검증한 `runs/hsx_port_comparison.json`이다. Raw trajectories, checkpoint, training log와 config를 함께 보관한다. Revised report의 source SHA256가 현재 source와 일치하며, baseline의 별도 source inventory도 보관한다.

## 완료 검증표

| 요구사항 | 현재 증거 |
| --- | --- |
| 선택한 encoder/scoring 구현 | `models/rxnflow.py`, block norm invariance 및 mixed-action gradient test |
| MSE, reward, replay, EMA, optimizer | `trainer.py`, `test_trajectory_balance_uses_backward_probability`, dropout·scheduler 포함 exact restart test |
| Mask, 종료, invalid, backward 연결 | `test_data_environment.py`, `test_failed_selected_action_is_retained_for_tb`, `test_retrosynthesis.py` |
| 1만 개 train/restart/sample | `./test.sh heavy`, gnode7 1 passed / 79.52s |
| 변경 전후 평가 | 두 evaluation.json, raw trajectory/count 검증, comparison.json |
| 문서와 리뷰 상태 | 본 문서, README, linear-synthesis, PLAN, PROGRESS; 사용자 리뷰 체크박스는 미완료 유지 |

Final quick는 35 passed, 1 heavy deselected / 9.50s이며 compile/lint/build/import를 포함한다. CUDA 실행은 이 CPU 결과로 검증했다고 주장하지 않는다. Production 준비 및 실험 화학 검증은 완료 범위에 포함하지 않는다.

## Batched execution (2026-10-02)

고정 graph shape를 실제 실행 경로에 적용했다. Rollout의 활성 state와 TB batch의 모든 transition을 각각 한 graph batch로 인코딩한다. State/reaction query head는 action 종류별로 묶어 호출하고, 선택된 block은 `(library, row)` 중복을 제거하여 fingerprint/property/type encoder를 함께 실행한다. Variable candidate 수는 flat tensor와 state별 split으로 처리한다. 같은 state의 graph는 call 안에서 재사용하지만, subsampling draw와 선택된 row의 budget mask는 state별로 계산하며 train dropout도 graph row마다 적용된다. Optimizer update를 넘는 learned embedding cache는 없다.

Uniform subsampling과 conditional inclusion correction은 유지하되, 전체 library 인덱스의 `randperm` 대신 checkpointed Torch RNG에서 seed를 받아 `random.sample(range(...))`로 필요한 개수만 뽑는다. Batch scheduling과 난수 소비 순서가 바뀌므로 이전 serial 실행과 분자별 동일 trajectory를 보장하지 않는다. 새 실행 경로 내 checkpoint 재시작은 기존 dropout/EMA/optimizer/RNG 재현 테스트로 검증한다.

전체 2,057,312-row 환경에서 gnode7 GPU 0(RTX 3080 Ti), hidden128/layers3/heads4, fresh32+replay32로 확인했다. 초기 replay 없는 update를 제외한 짧은 비교에서 serial median 43.69초/update → batched 8.86초/update(4.93배), rollout 9.49초 → 3.16초, peak allocated CUDA memory 5193.55 MiB → 3913.81 MiB였다. Serial 3 update와 batched 5 update를 같은 초기 checkpoint에서 시작했으며, RNG 소비 차이 때문에 동일 trajectory 비교는 아니다. 이는 처리 속도 확인이며 학습 성능 우위의 근거가 아니다. 로컬 기록은 `runs/full_stock_20261002_batched_gpu0/benchmark.json`에 보관한다.

## Mask 순서 정정 (2026-10-02)

`explore_250509`의 `policy/action_categorical.py::_calculate_logits`는 workflow/protocol이 정한 block type을 먼저 subsample하고 그 결과에 `_get_action_mask`를 적용한다. 앞서 전체 library mask → subsampling을 HSX와 동일한 순서로 설명한 것은 잘못이었다. 사용자의 지시에 따라 현재 구현은 가능한 reaction/type → library subsampling → sampled-row budget mask로 맞췄다. 이는 앞선 benchmark 이후의 변경이다. 기존 budget 규칙과 관측 action 강제 포함/conditional importance correction은 유지한다. 본 문서의 이전 30-update 및 batch 속도 비교는 당시 mask-first 실행의 역사적 결과이며 최신 순서의 성능 측정으로 해석하지 않는다.

## 현재 실행 경로의 후속 최적화

2026-10-02 전체 source 비교와 후속 변경은 [성능 검토 기록](performance-review.md)에 정리했다. 위 과거 실행 수치는 그 당시 snapshot의 결과이며, 후속 matrix scoring·mask batching·SDPA·I/O 변경의 실측 결과가 아니다.
