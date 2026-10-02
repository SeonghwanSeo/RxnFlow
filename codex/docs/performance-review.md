# 구현·성능 검토 — 2026-10-02

후속 수정: [기준 구현 복원 기록](implementation-deviations.md#restoration-implemented--2026-10-02)을 참고한다. 이 문서의 기존 실행 수치와 이전 구현 설명은 당시 snapshot에 해당한다. 후보 압축·관측 action 강제 포함·기존 attention 구조는 이후 제거했으며, 이 수치로 새 구현의 속도나 품질을 판단하지 않는다.


## 현재 milestone

Env·library, MDP·masking/backward, 모델·학습 구현과 1차 runtime 최적화를 완료했다. 사용자 코드 검토와 현재 full catalog의 장기·다중 seed 학습 및 품질 평가는 남아 있다. 학습은 중단 상태이며, 이전 전체 학습의 마지막 기록은 step318/checkpoint300이다. 짧은 pilot과 runtime 측정은 수렴·성능 우위 검증으로 간주하지 않는다.

후속 runtime 측정에서 gnode7 GPU0,64 fresh+64 replay,ratio0.01/min10 조건의 전체 step 중앙값은1.82초(워밍업 제외5회,1.74–1.98초), env 로딩은10.89초, peak Torch allocated memory는744MiB였다. 이 측정은 최소 library size 필터 적용 전의1,095개 library 환경 기준이다. 원시 기록은 로컬 `runs/runtime_d039792/`에 있다.

이후 `--min-library-size`를 추가했다(default1). Full catalog에는10을 적용해895개 library(35brick/860linker),2,056,409행을 유지하고200개 library/903행을 제외했다. 생존 행의 순서와 feature 값은 보존했다. 이전 checkpoint와는 library index가 달라 새 학습이 필요하다. Unfiltered env 백업은 사용자 요청으로 삭제했다. 최신 quick49개와 gnode7 filtered full-set CPU 학습·restart·sampling 통합1개가 통과했다. 필터 적용 후 GPU runtime은 아직 측정하지 않았다.

Property masking의 일괄 계산도 비교했다.158개 unique state,649개 library,19,234개 sampled row에서 유효 index가 동일했으며10회 중앙값은 library별93.9ms/일괄93.3ms였다. Gathering·comparison·nonzero 추출을 포함한 CPU 비교다. 일괄 계산은 조합을36만에서304만으로 늘리면서 의미 있는 속도 이득이 없어 기존 library별 구현을 유지했다. 기록은 로컬 `runs/property_mask_batch/`에 있다.

다음 순서는 사용자 코드 검토·필요한 리팩토링 후, 현재 catalog에서 새 학습을 시작하고 유효 생성률·reward·다양성/중복·경로 길이·seed 편차를 평가하는 것이다. 추가 production UniReaction의 수기 화학 검토는 별도 보류 항목이다.

## 당시 코드 감사 범위

현재 `src/rxnflow` 전체를 읽고, 로컬에 있는 아래 네 reference의 대응 경로를 비교했다. Reference checkout과 branch는 변경하지 않았다. 학습·benchmark·production env 재생성·heavy run은 실행하지 않았다. 아래 결과는 코드 검토 및 작은 synthetic fixture의 CPU quick 테스트에 근거하며, GPU 속도 향상 수치는 아직 없다.

| Reference | 확인한 commit | 주요 비교 대상 |
| --- | --- | --- |
| RxnFlow `master` / `origin/master` | `a39c7aebd45fddfdf72109ffa0f8e4b92f9c4458` | `models/gfn.py`, `policy/action_categorical.py`, `envs/retrosynthesis.py`, sampling/TB |
| CGFlow `main` | `89fe021304164764f4c027da2ee0c274de3065fa` | `models/gfn.py`, action subsampling/categorical, preparation, env/context |
| HSX `explore_250509` | `2d472443806ac107b9cdc8a65d03866302394d84` | `models/gfn.py`, `policy/action_categorical.py`, workflow sampling |
| HSX `main` | `e999c3d1b2f911d1b33fb8245c0a66a2946f14b0` | `models/gfn.py`, `models/layers.py:SimilarityMDP`, graph transformer |

## 변경한 경로

| 검토할 파일 | 기존 비용 | 수정 |
| --- | --- | --- |
| [policy.py](../../src/rxnflow/gflownet/policy.py) | 전체 후보 쌍의 hidden vector 두 벌을 gather한 뒤 elementwise 곱; 큰 global index `unique`와 library 전체 검색 | Reference처럼 library별 query × block matrix 계산. Sample row를 한 번 encode하고 유효 column만 gather. 전역 candidate embedding 복제 및 `unique` 제거. |
| [policy.py](../../src/rxnflow/gflownet/policy.py), [env.py](../../src/rxnflow/envs/env.py) | State/library마다 작은 budget mask 연산과 block feature gather 반복 | Library별로 서로 다른 state descriptor를 stack하여 mask를 한 번 broadcast. 같은 state는 mask와 CPU graph 재사용. |
| [policy.py](../../src/rxnflow/gflownet/policy.py) | State마다 작은 GPU `logsumexp` 호출 | Flattened logits의 segmented max/sum으로 안정적인 batch log probability 계산. 관측 row 및 conditional inclusion correction 유지. |
| [env.py](../../src/rxnflow/envs/env.py), [library.py](../../src/rxnflow/envs/library.py) | 매 state 전체 library 검색, type 문자열 split, action group 재생성 | Attachment type index, cached site types, `(site, final-step, may-terminate)`별 group 재사용. |
| [graph.py](../../src/rxnflow/envs/graph.py) | 원자·결합마다 작은 Torch tensor 생성/대입, graph와 mask가 descriptor를 각각 계산 | NumPy array에 직접 feature를 채운 뒤 tensor로 wrap. 한 번 계산한 descriptor를 graph와 mask가 공유. Feature 순서·차원 유지. |
| [policy.py](../../src/rxnflow/gflownet/policy.py) | Eval rollout에서 동일한 초기 graph를 batch 크기만큼 encode | Eval 모드에서는 unique state만 encode한 뒤 gather. Train 모드의 각 graph row는 독립 dropout 유지. 호출 사이 learned embedding cache 없음. |
| [graph_transformer.py](../../src/rxnflow/models/graph_transformer.py) | 직접 attention score/softmax/value 연산, 여러 identity mask 생성 | Native `scaled_dot_product_attention`과 한 번의 diagonal self-mask 설정. Learned edge bias와 dropout 의미 유지. |
| [rxnflow.py](../../src/rxnflow/models/rxnflow.py) | Block encoder 호출마다 property scale tensor 생성·전송 | Module buffer로 보관; checkpoint에 불필요하게 저장하지 않음. |
| [retrosynthesis.py](../../src/rxnflow/envs/retrosynthesis.py) | DFS의 precursor Mol→SMILES→Mol 반복, compatible library 순회 | Mol과 canonical key를 재귀에 함께 전달. Incoming/remaining type으로 library를 직접 조회. Signature를 노드마다 한 번 계산. |
| [env.py](../../src/rxnflow/envs/env.py) | Forward-only sampler도 전체 reverse catalog index 생성 | `retro_analyzer`를 첫 backward 사용 시 생성. |
| [replay.py](../../src/rxnflow/gflownet/replay.py) | 매 sample마다 deque 전체를 list로 복사 | List ring에 FIFO 저장하고 필요한 논리 index만 균등 추출. Checkpoint는 oldest-first로 저장해 restart 순서 유지. |
| [trainer.py](../../src/rxnflow/trainer.py) | `latest` 작성 시 checkpoint 전체 재로드·재직렬화; 마지막 step에서 중복 저장 | 직렬화된 파일을 복사하고, 마지막 step을 이미 저장했으면 재저장하지 않음. Loss graph 생성 전 gradient 정리. |
| [sampler.py](../../src/rxnflow/sampler.py) | `from_checkpoint`→constructor에서 두 번 로딩; 사용하지 않을 optimizer/replay까지 GPU로 이동 | Checkpoint를 CPU에서 한 번 읽고 필요한 model weights만 이동. 아래의 단일 constructor API 사용. |
| [synthon.py](../../src/rxnflow/envs/chemistry/synthon.py), [prepare.py](../../src/rxnflow/envs/prepare.py) | 이미 sanitize한 conversion product의 반복 파싱, 작업 batch마다 template compile | Canonical key와 Mol을 함께 전달. Template set은 worker process마다 한 번 compile. 원본 Mol은 수정하지 않음. |
| [prepare.py](../../src/rxnflow/envs/prepare.py), [features.py](../../src/rxnflow/envs/chemistry/features.py) | 모든 library 배열과 row별 임시 배열을 동시에 보관; 매 row Morgan generator 생성 | Library별 배열을 미리 할당하고 같은 NPZ schema로 순차 기록. Morgan generator를 process 내 재사용. |
| [cli/train.py](../../src/rxnflow/cli/train.py) | 종료 시 worker pool 정리가 명시적이지 않음 | `finally`에서 생성된 reverse worker 종료. |

## 유지한 동작과 의도적인 차이

기본값은 fresh 64 + replay 64, sampling ratio 0.01, minimum 10이다. Library별 batch 공통 draw → state별 additive budget mask 순서를 유지한다. TB/replay는 관측 row의 합집합을 포함하며, forced row는 inclusion 1, 나머지는 conditional inclusion probability를 사용한다. 후보 전체를 action 객체로 만드는 경로는 앞선 수정에서 제거되어 있고, 이번 변경에도 다시 추가하지 않았다.

Workflow, action clustering/tier, reaction-first hierarchical distribution, HSX 전용 Synple magic number는 가져오지 않았다. 현재 합의한 선형 synthon MDP와 brick/linker orientation, UniReaction 종료, MSE TB, block-only normalization과 reaction temperature를 유지한다. Reverse search의 depth pruning·canonical precursor 최대 두 개·known branch 보존·forward 확인도 유지한다. Raw RDKit matches를 먼저 잘라 속도를 높이는 것은 선택되는 canonical precursor를 바꿀 수 있으므로 하지 않았다.

Reference의 PyG/scatter 의존성을 추가하지 않았다. Fixed-capacity graph와 native Torch를 사용한다. Library별 matrix 크기, GPU kernel launch 비용, native attention backend 선택 및 실제 VRAM/throughput은 GPU run 없이 결론 내릴 수 없다. 이번 작업 전의 benchmark 수치를 새 코드의 결과로 해석하지 않는다.

Config, reward API, serialization types, CLI entry points도 검토했다. Reward 입력·출력 검증은 외부에서 주입하는 reward의 계약 확인이므로 유지한다. 새 반응 결과의 화학적 타당성·capacity 검사는 prepared catalog의 중복 검증과 다르므로 유지한다. Sampling의 CPU RNG 경로도 checkpoint 재현 규칙을 유지하며 사용한다.

## Python sampler API

```python
from rxnflow.sampler import RxnFlowSampler

sampler = RxnFlowSampler("checkpoint.pt", reward=my_reward, device="cpu")
results = sampler.sample(64, seed=0)
```

`RxnFlowSampler(config, checkpoint, ...)`와 `from_checkpoint(...)`를 없애고 checkpoint에서 config를 읽는 constructor 하나로 정리했다. CLI도 같은 경로를 사용한다. 호환 alias는 추가하지 않았다. 기존 prepared env의 NPZ key/shape/dtype은 그대로이며 재생성이 필요 없다. 이미 실행한 run의 source snapshot·checkpoint는 수정하지 않았다.

## 검증

`OMP_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-audit-final-quick.log 2>&1` — Python 3.10 compile/lint/build/import 성공, **48 passed, 1 heavy deselected, 9.47s**. `.venv/bin/ruff check src tests codex/scripts`, `git diff --check` 성공.

테스트는 작은 synthetic fixture만 사용했다. Full/partial subsampling에서 matrix/scalar score와 parameter gradient, segmented/scalar log probability 및 gradient, native/explicit attention 출력 및 gradient, scalar/batch budget mask, FIFO wrap 및 restart, single CPU checkpoint load, serial/parallel preparation 배열 일치, dropout·EMA·optimizer·RNG 포함 restart를 확인했다. Production training, GPU benchmark, heavy test는 사용자 요청대로 실행하지 않았다.

검토 순서는 policy의 matrix/mask 부분 → graph/attention → env/reverse lookup → replay/checkpoint/sampler → prepare streaming을 권한다. 기존 working tree 변경은 그대로 유지했으며 이번 검토를 핑계로 무관한 파일을 commit하지 않았다.
