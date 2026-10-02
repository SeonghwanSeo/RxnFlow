# 현재 구현과 네 reference의 섹션별 비교

작성일: 2026-10-02. 비교는 중간 commit `2aac819`에서 시작했으며, 아래 현재 구현에는 이후 승인된 bond stereo·temperature 초기값·mask margin·block projection·MLP/embedding 초기화·모델 크기·clip/random 기본값 변경도 반영했다. 현재 구현은 **HSX main의 일괄 이식이 아니라, HSX 250509의 graph readout/conditioning·fusion MLP 순서 + HSX main의 block projection/similarity/TB/optimizer 구성 + RxnFlow master·CGFlow의 categorical/subsampling/backward를 결합한 구현**이다. Enamine 환경과 workflow 없는 선형 MDP는 별도로 작성했다.

후속 사용자 결정: property scale은 사용자가 단순한 값으로 바꾼 것이므로 유지한다. Bond stereo는 main의 categorical, temperature 초기값은 0.2, mask margin은 1%로 변경했다. FP/property projection은 main의 Linear→LayerNorm으로 단순화했다. MLP output은 Xavier, reaction/type embedding은 uniform[-0.1,0.1], graph 크기는 128/2heads/4layers, block_dim은 128로 맞췄다. Clip은 100, random은 0.1, reward floor는 1e-4다. Fusion/policy의 normalization 순서, 2H readout, GNN 이후 reaction embedding은 기존 250509 방식을 유지한다. GENConv bias와 empty state는 사용자가 추후 architecture 검토까지 보류했다. MW는 후속 사용자 선택으로 Descriptors.ExactMolWt로 변경했다. Atom feature는 아래 분석만 수행하고 변경하지 않았다.

이 문서는 소스와 기본 설정을 비교한 결과다. 동일하다는 판정은 명시한 수식 또는 동작 범위에 한정한다. Reference 네 개를 동일 데이터로 학습해 출력·성능을 비교한 결과가 아니며, 네 checkpoint와의 호환성을 뜻하지 않는다. 특히 “reference graph equations 복원”은 HSX main 전체 모델과의 동일성을 의미하지 않는다.

## 1. 비교 기준 고정

| 표기 | 사용자 명칭 | 저장소와 ref | 읽은 commit |
| --- | --- | --- | --- |
| 현재 | 현재 개발 구현 | 이 저장소 `version/hsx` | `2aac819` + 위 후속 승인 변경 |
| R | rxnflow master | 이 저장소 `master` | `a39c7aebd45fddfdf72109ffa0f8e4b92f9c4458` |
| C | cgflow-rxnflow | `source/CGFlow`의 `src/rxnflow` 및 연결된 GFlowNet 코드 | `89fe021304164764f4c027da2ee0c274de3065fa` |
| H | hsx | `source/rxnflow_hits`의 `main` | `e999c3d1b2f911d1b33fb8245c0a66a2946f14b0` |
| X | hsx-250509 | `source/rxnflow_hits`의 `explore_250509` | `2d472443806ac107b9cdc8a65d03866302394d84` |

Reference는 위 commit의 파일을 `git show`로 읽었다. HSX 작업 디렉터리는 250509이므로, 작업 디렉터리의 같은 파일명을 main의 근거로 사용하지 않았다. C의 비교 범위는 RxnFlow 생성 경로이며, 별도 3D 모델·docking 구현은 포함하지 않는다. 아래 “reference”는 코드상 대응과 기존 이식 기록으로 확인되는 기준이다. 비슷한 코드가 존재한다는 이유만으로 실제 복사 이력을 단정하지 않는다.

## 2. 현재 구현의 출처와 HSX main 일치 여부

| 섹션 | 현재 기준 | HSX main과의 관계 |
| --- | --- | --- |
| Enamine 준비·library·source BB mapping | 현재 환경 요구사항에 맞춘 별도 구현 | 다름. Synple, tier, cluster, workflow 산출물 없음 |
| State/action/종료 | C의 typed synthon 선형 조립과 유사한 틀 + 현재 UniReaction 계약 | 다름. Workflow/order 대신 현재 site와 reaction 수로 결정 |
| Fingerprint | H의 count Morgan 512 계열 + dummy isotope invariant·uint8 | 일부만 같음. 입력, invariant, 순서, dtype 다름 |
| Atom/bond/global features | 현재 synthon 환경용 정의 + H bond stereo | Bond type/stereo/flags는 H와 같음. Atom vocabulary와 사용자 지정 scale은 다름 |
| Property budget | X의 sampled-row 적용 + H의 1% tolerance | Positive bound 판정은 H와 같음. Sampling과 all-masked 처리는 다름 |
| Graph message passing | R/C/X의 GENConv(add) + TransformerConv를 native Torch로 구현 | 같은 계열. H의 GENConv bias 설정은 다름 |
| Graph readout·reaction conditioning | X 중심, R/C와도 공통 | 다름. H의 2H→H projection 및 GNN 이전 action conditioning 없음 |
| Block encoder·policy MLP | H의 feature projection/초기화 + X의 fusion/policy 순서, tier 제외 | FP/property의 Linear→LN, block_dim128, output Xavier와 embedding 초기화는 H와 같음. Fusion 깊이·normalization 순서 및 policy input 차원은 다름 |
| Block score | H의 SimilarityMDP(dot) 수식 | Action만 L2 normalize하는 수식·temperature 범위·초기값은 같음. Parameter 구분은 다름 |
| UniReaction score | R의 learned scalar head 계열 + 현재 typed MDP | 다름. H/X에서는 workflow가 선택하므로 logP=0 |
| Subsampling·categorical·observed logP | R/C + X의 budget mask | 다름. H는 cluster→block 계층 정책 |
| Batch 안의 library draw 공유 | 사용자 요청에 따른 현재 구현 | 다름. C보다도 공유 범위가 넓음 |
| Backward·reverse worker overlap | R 중심 | 다름. H/X의 backward logP는 0 |
| TB residual·scalar logZ | H | MSE residual과 scalar parameter 형태는 같음. PB·reward 기본값은 다름 |
| AdamW·두 learning rate·EMA | H의 구성 | 구성·EMA 수식과 clip100은 같음. WD/EMA 기본값 다름 |
| FIFO replay | H의 비복원 추출 | 추출 원리는 같음. 저장 자료구조·warmup·기본 활성화 다름 |
| Checkpoint·local reward·API | 현재 공개 범위에 맞춘 구현; X에도 full restart 선례 있음 | Artifact/API는 다름 |

“다름”에는 합의된 변경과 아직 정렬하지 않은 세부사항이 함께 있다. 둘은 8절에서 구분한다. 최초 분석과 후속 구현의 검증 기록은 9절에서 구분한다.

## 3. 환경, 표현, feature

### 3.1 환경 및 MDP

| 부분 | R | C | H | X | 현재 구현과 기준 |
| --- | --- | --- | --- | --- | --- |
| State | 실제 분자, NetworkX 기반 MolGraph | Typed synthon, NetworkX 기반 MolGraph | RDKit Mol/SMILES + workflow/traj_idx | NetworkX 기반 MolGraph + workflow/traj_idx | RDKit Mol + reaction_count + terminated. Tensor graph는 별도 입력 표현. H와 Mol 보유 방식은 유사하나 상태 의미가 다름 |
| Library | Global BB 목록 + reaction별 적합 mask | Type별 brick/linker library | Synple/eXplore type + tier + cluster | Synple/eXplore type + tier | Enamine 변환 후 type별 oriented brick/linker, 중복 synthon의 source ID 집계. 직접 대응하는 prepare 구현 없음 |
| Prepared feature | FP/descriptor NPY 및 BB mask | `bb_feature.pt`와 type별 SMILES | FP/property/cluster 등의 NPZ 산출물 | Type별 feature/tier 산출물 | 단일 `bb_feature.npz`, provenance JSON, aligned SMILES. 파일 형식은 현재 독자 정의 |
| 반응 정의 | Reaction SMARTS에서 reverse 생성 | YAML에 forward/reverse 명시 | Protocol YAML + workflow CSV | Protocol YAML + workflow CSV | `synthon.yaml` 변환과 `reaction.yaml` forward/reverse 분리. C의 명시적 reverse와 가까움 |
| 다음 action | Stop/First/Uni/Bi, reactant SMARTS 적합성 | Site type에 맞는 First/Bi | Workflow의 현재 protocol | Workflow의 현재 protocol | First/Uni/Bi 중 typed site 및 reaction 수로 결정. Workflow/Stop 없음 |
| Linker 방향 | BB/reactant order 처리 | Type별 synthon/protocol 처리 | Workflow/protocol에 종속 | Workflow/protocol에 종속 | Incoming site `[*]`를 고정한 양방향 row. Block 선택이 방향 선택을 포함하는 현재 계약 |
| UniReaction | 실제 분자 SMARTS 변환을 action으로 선택 | 해당 Workflow loader는 Uni를 지원하지 않음 | 정해진 workflow step | 정해진 workflow step | Typed site와 이웃 substructure에 작용. Protected type→active type 또는 terminal 생성. 독립 learned action |
| 종료 | Stop, 길이 제한 | Site 소진, 마지막 step linker 제외 | Workflow 끝 | Workflow 끝 | Brick 결합 또는 terminal Uni. 최대 reaction에서는 terminal action만 허용. 최소 이전에는 terminal action 제외 |
| 선택 후 처리 | 선택된 반응 실행 | 선택된 반응 실행 | 선택된 protocol 실행 | 선택된 protocol 실행 | 선택된 RDKit 반응만 실행. Canonical 중복을 합친 뒤 connected product 하나만 허용, site signature와 실제 max_atoms 확인 |

C의 `Workflow`라는 클래스명은 이 버전에서 protocol catalog를 읽는 역할이다. H/X처럼 trajectory에 workflow ID와 step 순서를 고정하는 것과 구분해야 한다. 현재는 상태당 최대 한 dummy, brick당 한 site, linker당 두 site라는 계약이므로 incoming marker를 고정한 library row로 위치 선택을 처리한다. 임의의 multi-site 분자를 그대로 지원하는 구현은 아니다.

현재 property masking은 **실제 후보 생성물을 만들지 않는다**. Selected reaction 이후에도 정확한 생성물 property 상한을 검사하지 않고, 실제 heavy-atom capacity와 구조 계약을 검사한다. Uni에는 추가 block이 없으므로 additive block budget을 적용하지 않는다. 이는 reaction SMARTS를 확인하지 않고 모든 Uni가 성공한다는 뜻은 아니며, 선택 후 substructure match 실패는 invalid trajectory가 된다.

근거: 현재 [env.py](../../src/rxnflow/envs/env.py)의 `available_groups`, `_groups_for`, `block_mask`, `step`; [prepare.py](../../src/rxnflow/envs/prepare.py)의 `convert_stage`, `features_stage`; [library.py](../../src/rxnflow/envs/library.py)의 `load_block_libraries`; [reaction.py](../../src/rxnflow/envs/chemistry/reaction.py)의 `Reaction.run_forward`. Reference: R/C/H/X의 `src/rxnflow/envs/env.py`, C/H/X의 `envs/workflow.py`, 각 `envs/env_context.py`.

### 3.2 모델 입력 feature

| 부분 | R | C | H | X | 현재 구현 |
| --- | --- | --- | --- | --- | --- |
| FP | Binary Morgan 1024 + MACCS | Binary Morgan 1024 + MACCS | Count Morgan 512 + MACCS | Binary Morgan 1024 + MACCS | Count Morgan 512 + MACCS. Radius 2는 공통 |
| FP 저장 | bool | bool | float16 | bool | Count를 255에서 clamp한 uint8, 선택 row만 float32 변환 |
| FP 순서 | MACCS→Morgan | MACCS→Morgan | MACCS→Morgan | MACCS→Morgan | Morgan→MACCS |
| MACCS slice | `[:166]` | `[:166]` | `[1:167]` | `[1:167]` | `[1:]`, 실제 166 keys. R/C와 동일하지 않음 |
| Site label FP | 기본 Morgan invariant | 기본 Morgan invariant | At 기반 synthon, 기본 invariant | At 기반 synthon, 기본 invariant | Dummy isotope를 custom atom invariant로 명시. Incoming `[*]`와 남은 typed dummy를 구별 |
| Block property | 8종, MW/HBA/HBD 등 | 8종, R과 같은 구성 | 9종, ring/arom/rotb 포함 | H와 같은 9종 계열 | 9종, 항목은 H/X와 대응하지만 순서·scale·일부 함수 다름 |
| MW | ExactMolWt | ExactMolWt | ExactMolWt + Synple 보정 | ExactMolWt + Synple 보정 | Descriptors.ExactMolWt, dummy isotope만 0으로 만든 뒤 계산. H와 같은 exact mass 정의, Synple 보정 제외 |
| Atom | Element/charge/H/chirality/aromatic categorical | Element/charge/H/chirality 및 synthon isotope | Element/total degree/charge/chirality/H/hybridization/isotope 등의 vocabulary | Element/charge/explicit H/chirality/isotope/aromatic | Atomic number, degree, charge, dummy type one-hot + aromatic/mass/total H/hybridization scalar + chirality 3종 |
| Bond | Bond type 4종 | Bond type 4종 | Bond type + conjugation + ring + stereo categorical | Bond type 4종 | H와 같은 type 4개 + stereo 7개 + conjugation/ring 2개, 총 13차원 |
| Global input | 분자 property + 외부 conditional 정보 | 분자 property + 외부 conditional 정보 | State property + workflow/order action embedding | 분자 property + 외부 conditional 정보 | 9 property + remaining capacity + reaction-count one-hot. Beta conditional 없음 |
| Graph batch | PyG sparse | PyG sparse | PyG sparse | PyG sparse | `max_atoms + 1` molecular slots의 fixed tensors + node mask, model 내부 virtual node. Heavy atoms를 truncate하지 않음 |

현재 property 순서는 `mw, tpsa, hbd, hba, logp, rotatable_bonds, rings, aromatic_rings, heavy_atoms`, scale은 각각 `100, 100, 10, 10, 10, 10, 10, 10, 100`이다. H/X의 순서는 `heavyatom, hba, hbd, ring, arom, rotb, mw, tpsa, logp`, scale은 `10, 5, 5, 10, 5, 5, 500, 50, 5`다. 현재 scale은 사용자가 단순한 값으로 조정한 선택이며 이식 누락으로 분류하지 않는다. Scale은 neural input에만 적용하고 mask는 raw 단위로 비교한다. 별개인 MW 함수 정의는 budget 판정도 바꿀 수 있다.

H의 block property에는 At isotope 질량 차감과 linker `+29.0`이 있다. X도 같은 종류의 block 보정을 사용한다. H는 state에서도 At isotope 질량을 빼지만 X의 `get_mol_properties`는 그 별도 state 보정을 하지 않는다. 현재 Enamine dummy에는 이 Synple 상수를 적용하지 않는 것이 기존 합의다. 다만 **MolWt 선택과 normalization scale 차이까지 Synple 상수 제거의 필수 결과인 것은 아니다**.

H의 bond stereo는 NONE/ANY/Z/E/CIS/TRANS와 unknown을 구분한다. 최초 비교 대상 `2aac819`는 presence bit만 사용했으나, 후속 요청에 따라 같은 categorical을 적용했다. E/Z/unspecified의 node features와 connectivity가 같아도 bond features 및 GNN 출력은 구분된다. 일반 원자의 isotope는 여전히 H/X처럼 별도 categorical로 넣지 않으며, 현재 mass scalar에 영향을 주는 정도다. Dummy isotope만 별도 one-hot으로 넣는다.

Atom feature 차이는 저장 형식뿐 아니라 정보 정의·범위와 첫 projection의 inductive bias에도 있다. H count의 scalar는 count 자체를 보존하면서 차원을 줄인다. Hybridization의 enum scalar도 현재 범위에서는 type을 구별하지만, SP/SP2/SP3 사이에 임의의 수치적 순서를 부여한다. Categorical은 각 type에 독립적인 weight를 학습하게 한다. 또 현재 `GetDegree()`와 H의 `GetTotalDegree()`는 H를 포함하는 방식이 다르고, 일반 isotope categorical과 mass scalar도 동일한 입력이 아니다. 따라서 H count scalar는 유지할 만하고 hybridization categorical은 추후 검토할 만하지만, 이번에는 atom feature를 변경하지 않았다.

Dense float32 node input은 현재 222차원, H 기본 vocabulary에서는 152차원이다. 현재의 element/synthon one-hot만 101+101차원이므로, 일부 scalar를 쓴다는 이유로 전체 feature가 더 작지는 않다. 같은 node 수와 hidden width라면 152차원은 입력 저장과 첫 Linear의 곱셈 수를 약 31.5% 줄인다. 이후 GNN hidden 연산량까지 같은 비율로 줄어드는 것은 아니며 실제 runtime을 측정한 결과도 아니다. H vocabulary는 At 및 Synple/eXplore isotope용이므로 Enamine `*`와 type에 그대로 적용할 수 없다. 비용을 줄이려면 지원할 element/type vocabulary부터 결정하는 편이 효과적이다.

Empty graph도 완전히 같지 않다. Reference의 `graph_to_Data`는 empty용 node를 만들고 backbone이 virtual node를 추가한다. 현재는 molecular node가 0개인 mask와 virtual node만 사용한다. 따라서 FirstBlock의 초기 graph embedding까지 수치적으로 동일하다고 주장하지 않는다.

근거: 현재 [features.py](../../src/rxnflow/envs/chemistry/features.py)의 `molecular_properties`, `block_fingerprint`, [graph.py](../../src/rxnflow/envs/graph.py)의 `molecule_to_graph_data`. R `envs/building_block.py`, C `utils/featurization.py` 및 `envs/env_context.py`, H `utils/feature.py`, `utils/vocab.py`, X `envs/building_block.py`, `envs/env_context.py`. 경로는 별도 표시가 없으면 해당 ref의 `src/rxnflow/` 아래다.

## 4. GNN, block encoder, scoring

| 부분 | R | C | H | X | 현재 구현과 기준 |
| --- | --- | --- | --- | --- | --- |
| Message passing | GENConv(add) + TransformerConv | 동일 계열 | 동일 계열 | 동일 계열 | R/C/X의 pre-norm, residual, conditional scale/shift, virtual edge/self-loop 계산을 native Torch로 구현 |
| GENConv bias | Constructor에서 미지정 | 미지정 | Helper에서 `bias=True` 강제 | 미지정 | `bias=False`. 현대 PyG 기본값을 따르는 R/C/X 쪽. H와 다름 |
| Attention head | Head마다 H channel, concat | 동일 계열 | 동일 계열 | 동일 계열 | Head마다 H channel. H를 head 수로 나누는 축소형 attention이 아님 |
| Normalization | Graph-mode LayerNorm + 조건부 affine | 동일 계열 | 동일 계열 | 동일 계열 | Graph 전체 valid node/channel로 정규화. Padding 제외, virtual 포함 |
| Readout | Molecular mean + virtual, 2H; model LN | 동일 계열 | 2H concat→Linear→H→LN | 2H concat + LN | X/R/C 계열의 2H, 추가 projection 없음 |
| Action conditioning | GNN 이후 reaction embedding + SiLU, FirstBlock은 별도 | GNN 이후 protocol embedding + SiLU | Workflow/order embedding을 GNN condition에 포함 | GNN 이후 workflow/order embedding + SiLU | X의 위치를 따름. ID는 reaction name이고 FirstBlock에도 embedding 적용 |
| FP/property projection | 각각 single Linear→LN→activation | 각각 single Linear→LN→activation | 각각 single Linear→LN | 각각 Linear→SiLU→Linear→LN→SiLU | H와 같은 single Linear→LN으로 변경 |
| Block fusion | FP+property | FP+property+type | FP+property+type+tier | FP+property+type+tier | X에서 tier 제외, 3×block_dim concat→MLP |
| Block logits | Raw dot + conditional logit scale | Raw dot + conditional logit scale | SimilarityMDP, cluster와 block 두 단계 | Raw dot + conditional logit scale | H의 action-normalized dot을 flat sampled block policy에 적용 |
| Uni logit | Learned scalar | 해당 loader에서 지원 안 함 | Workflow로 고정, logP=0 | Workflow로 고정, logP=0 | Learned scalar / reaction temperature. R과 유사하나 현재 site/action 정의에 맞게 확장 |
| logZ | Conditional MLP | Conditional MLP | Scalar parameter | Conditional MLP | H와 같은 scalar, 초기값 0 |

현재 block score는 `q_r(s) · normalize(e(b)) / T_r`이며 `T_r = 0.01 + 9.99 × sigmoid(t_r)`이다. Query를 normalize하지 않는 것은 H의 dot 모드와 같다. 그러나 H는 workflow/order별 cluster score와 block score에 각각 SimilarityMDP를 두고, 현재는 reaction별 temperature를 둔다. H의 `SimilarityMDP` constructor 기본 인자는 1이지만 **실제 ModelConfig 기본값은 0.2**를 전달한다. 후속 요청으로 현재 초기값도 1에서 0.2로 변경했다.

FP/property projection은 H처럼 Xavier weight와 zero bias로 초기화한다. 후속 요청으로 graph/fusion/policy MLP의 activation 없는 output도 Xavier로, reaction/type embedding도 uniform[-0.1,0.1]로 맞췄다. Hidden layer는 graph의 LeakyReLU Kaiming, fusion/policy의 SiLU용 Kaiming을 유지한다. Fusion/policy hidden 순서는 현재 `Linear→LN→SiLU`이고 H는 `Linear→SiLU→LN`이다. Norm 뒤의 SiLU는 입력 scale을 정리하고 activation을 적용하며, SiLU 뒤의 norm은 activation 출력을 다시 중심화한다. 두 연산은 같지 않다. [Torchvision MLP](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/misc.html)도 Linear→norm→activation을 사용하므로 현재 순서는 conventional한 선택이며, 성능 우위를 확인하지 않고 순서만 바꾸지 않는 것을 권고한다. 이 순서는 이번에 변경하지 않았다.

R/C/X는 GENConv의 bias를 명시하지 않으므로 원래 설치된 PyG 버전에 따라 결과가 달라진다. H의 `_gen_conv_kwargs`는 이 문제를 피하려고 True를 명시한다. 현재 `False` 구현에 대한 독립 수식·gradient test는 통과했지만, H의 bias=True graph를 그대로 재현한 테스트는 아니다. 또 native nn.Linear와 reference 내부 Linear의 초기화까지 모든 parameter가 같다는 검증은 하지 않았다.

근거: 현재 [graph_transformer.py](../../src/rxnflow/models/graph_transformer.py):17,33,46 및 [rxnflow.py](../../src/rxnflow/models/rxnflow.py):22,40. R/C/X `src/gflownet/models/graph_transformer.py`, 각 `src/rxnflow/models/gfn.py`; H `src/rxnflow/models/graph_transformer.py`:151, `models/layers.py`:44,94, `models/gfn.py`:100, `models/nn.py`:242, `models/config.py`:71.

### 구조 변경에 대한 해석

H의 encoder/readout 개편은 `a1afab8` (`feat: v1.3.0: ES reward, BB clust, pixi (#98)`)의 clustering 개편과 함께 들어갔다. 다음은 코드 구조에 대한 해석이며 당시 사용자의 의도를 단정하거나 성능 우위를 주장하는 것은 아니다.

- H는 FP/property별 projection을 얕게 하고 합친 feature를 fusion MLP에서 처리한다. X는 각각의 feature에 비선형 변환을 한 번 더 적용한 뒤 합친다. H도 fusion MLP가 있으므로 비선형 표현력이 사라지는 것은 아니다. 동일 width에서는 branch 연산이 줄지만, 이후 현재 block width도 64→128로 키웠으므로 이전 모델보다 전체 비용이 줄었다고 단정할 수 없다.
- H의 readout projection은 molecular mean/virtual node를 공통 embedding 차원으로 섞어 cluster와 block에 직접 similarity score를 계산하도록 한다. X/현재는 두 정보를 2H로 넘기고 reaction-conditioned policy head에서 섞는다. 두 방식은 압축과 reaction별 처리의 위치가 다르며, projection 추가 자체가 성능 향상을 보장하지 않는다.
- H의 early conditioning은 reaction/workflow에 따라 message passing부터 달라질 수 있다. 대신 같은 분자라도 condition이 달라지면 GNN 출력도 다시 계산해야 한다. 현재는 한 state에서 여러 reaction을 비교하므로 post-GNN conditioning을 유지해 graph encoding을 공유한다. 이는 사용자가 명시적으로 유지하기로 한 선택이다.

후속 사용자 요청으로 block의 FP/property projection을 H의 Linear→LN으로 단순화하고, 다음 phase에서 block_dim128과 H의 output/embedding 초기화를 적용했다. Fusion MLP의 비선형 처리와 한 hidden layer, late conditioning 및 2H readout은 유지한다. 따라서 H의 전체 모델을 그대로 이식한 것이 아니며 단순화 자체로 성능 우위를 주장하지 않는다.

## 5. Masking, subsampling, log probability, 실행

| 부분 | R | C | H | X | 현재 구현과 기준 |
| --- | --- | --- | --- | --- | --- |
| Chemical eligibility | Reactant SMARTS, Stop 조건 | Site type 및 protocol/길이 | Workflow에서 정해진 protocol | Workflow에서 정해진 protocol | Type/reaction count로 group 결정, 선택된 반응의 SMARTS는 실행 시 확인 |
| Property mask | 이 policy 경로에 HSX식 additive mask 없음 | 이 policy 경로에 HSX식 additive mask 없음 | State+block normalized budget `<1.01` | State+block normalized budget `<1.001` | Margin은 H 기준으로 변경. Raw 단위 비교로 zero/negative bound 지원, heavy atoms strict |
| Action 축소 | Protocol별 uniform subsampling | Protocol 내 library별 uniform subsampling | Cluster→그 cluster의 block 선택 | Block type 내 tier별 stratified subsampling | Uniform library draw. Tier/cluster 없음 |
| Draw 공유 | Protocol draw를 batch에 공유 | Protocol의 library draw를 batch에 공유 | 해당 없음 | State별 draw | Library별 한 번 뽑아 states와 reactions 모두 공유. C보다 넓은 공유는 사용자 요청 |
| Linker draw rate | 별도 현재식 library 구분 없음 | Linker type 수에 따른 감소 계수 | Subsampling 아님 | Tier 분할 기준 | 모든 library에 ratio 0.01, min 10. C의 linker 보정 계수 없음 |
| 전체 library 선택 | 작은 space는 arange | 작은 space는 arange | 선택 cluster 내 전체 blocks | Tier별 크기에 따라 full/partial | `n=min(N,max(10,floor(0.01N)))`; n=N이면 cached indices, RNG 없음 |
| 후보 score 형식 | Protocol별 matrix | Protocol에 속한 library들을 concat한 matrix | Cluster logits 후 선택 cluster의 block logits | State별 vector, workflow가 protocol 제한 | Reaction별 eligible state rows × sampled columns. Mask된 열도 유지하고 -inf |
| Importance correction | log(N/n)로 denominator 추정 | Library별 log(N/n) | Hierarchical exact softmax | Tier별 inclusion correction | R/C. Mask 생존 개수로 N/n을 다시 계산하지 않음 |
| Observed action | Numerator 별도 계산, denominator 독립 draw, logP≤0 clamp | 같은 구성 | Hierarchical selected cluster/block logP | 별도 numerator + subsampled denominator | R/C 계열. Observed action을 draw에 강제로 넣지 않음 |
| Sampling RNG | Device Gumbel | Device Gumbel | Softmax + multinomial | State별 device Gumbel | R/C의 matrix Gumbel. 선택 좌표만 CPU로 이동 |
| Random policy | Protocol별 sampled 수 보정 | `-log(n_libraries × n_sampled_library)` | Cluster/block 각 단계 random, novelty 경로도 있음 | 현재 workflow protocol 내 random | C의 offset. Mask 후 생존 block 전체에 단순 uniform인 것은 아님 |
| All masked | 주로 protocol eligibility mask | 주로 protocol eligibility mask | Cluster/block logits를 완화하는 fallback 경로 있음 | Invalid logit을 finite log epsilon으로 둠 | -inf mask 유지, rollout 실패. Denominator 계산에만 finite floor 사용 |
| Reverse overlap | 다음 forward 후 이전 reverse 수집 | 같은 pipeline 계열 | 이 reverse 계산 없음 | 이 reverse 계산 없음 | R 기준. 마지막 pending 결과도 수집 |

현재 순서는 **호환 reaction/library 결정 → library 공통 subsampling → sampled rows에 state+block mask → logits matrix → 선택된 action 객체 생성**이다. 모든 block에 property mask를 먼저 적용한 뒤 그 생존 집합에서 sampling하는 방식은 아니다. 이는 X의 sampled-row budget 적용을 따른 것이며, 지난 대화에서 서로 다른 의미의 “mask 이후 sampling”을 혼용한 부분을 이 순서로 고정한다. Subsample에 feasible block이 없으면 refill하지 않는다.

“Indexing을 한다”는 표현도 나누어야 한다. Reference와 현재 모두 library row 추출, reaction을 허용하는 state row 선택, selected logit 추출에는 indexing을 쓴다. 제거한 것은 **property mask의 True 좌표마다 후보 edge를 만들어 압축·재조립하는 처리**다. 현재도 protocol/library 반복은 있지만 block encoder는 필요한 sampled features를 합쳐 한 번 호출하고, score GEMM은 reaction별로 합친 library에 수행한다. 전체 reaction을 하나의 거대한 GEMM으로 만들지는 않는다.

현재 denominator는 sampled valid logits에 inclusion weight를 적용한 합이다. `log P = min(observed_logit - log(estimated_partition), 0)`이라는 R/C의 추정식은 유지한다. 그러나 빈 합의 floor 위치까지 완전히 같지는 않다. 현재는 최종 합을 `1e-38`에서 clamp하고, reference categorical들은 protocol/entry 단위 epsilon 처리가 다르다. 따라서 fully masked 또는 극단적 underflow에서 같은 gradient를 보장하지 않는다. H의 full hierarchical softmax와도 다른 추정량이다.

H의 property constraint에는 이름이 비슷한 두 함수가 있다. Action mask는 `models/gfn.py:_get_action_mask`의 1% tolerance다. `envs/env_context.py:check_property_constraint`에는 별도 0.1% 값이 있지만, 조사한 H source에서 호출 지점을 찾지 못했다. 후자를 근거로 “HSX main action mask도 0.1%” 또는 “main이 항상 실제 생성물을 검사한다”고 판단하지 않았다.

Library feature storage/cache도 같지 않다. H context는 block feature/budget/cluster 정보를 device cache에 준비한다. 현재는 CPU uint8 FP와 float32 property를 유지하고 sampled rows만 device로 보내며, update 간 학습된 embedding을 캐시하지 않는다. 동일 state의 CPU graph/property 구성은 재사용하고 eval에서 graph encoding도 공유하지만 training은 별도 graph occurrence를 유지한다. 이는 현재의 메모리·batch 실행 선택이며 HSX main cache 구현의 복제는 아니다.

근거: 현재 [policy.py](../../src/rxnflow/gflownet/policy.py):59,182,256,274, [categorical.py](../../src/rxnflow/gflownet/categorical.py):47,60, [subsampling.py](../../src/rxnflow/gflownet/subsampling.py):22. R/C/X `src/rxnflow/policy/action_categorical.py`, `action_space_subsampling.py`; H `src/rxnflow/models/gfn.py`:283,353,775,846; R/C `src/rxnflow/algo/synthetic_path_sampling.py`.

## 6. Backward, loss, optimizer, replay, checkpoint

| 부분 | R | C | H | X | 현재 구현과 기준 |
| --- | --- | --- | --- | --- | --- |
| Backward probability | Reverse tree, Σ A^(-depth), A=action-space 크기 | 유사 reverse tree, 고정 A=10000 | Log PB=0 | Log PB=0 | R의 깊이 가중 방식. 현재 action space에 맞춘 A 사용 |
| Reverse search | Depth/approximation 및 cache, 최소 depth pruning | 유사 근사 search | Workflow route | Workflow route | Depth bound 안의 canonical precursor + catalog match + forward 확인, 생성 route 보존. Reference search 그대로는 아님 |
| Backward 분자 | Action matching | Action matching | 해당 없음 | 해당 없음 | Action과 parent SMILES 모두 일치하는 branch 합산 |
| TB | Conditional logZ, 여러 TB 옵션 | 같은 generic GFN 계열 | Scalar logZ, 기본 MSE, MAE/Huber 옵션 | Conditional logZ, generic TB 옵션 | H의 scalar MSE TB를 직접 구현. R식 nonzero PB를 사용 |
| Reward | Task/conditional framework | Task/conditional framework | Reward function 및 client stack | Task/conditional framework | Local RewardFunction, beta×log(max(R,floor)). QED+Lipinski는 QED reward와 MW/HBA/HBD mask |
| Optimizer | Adam, policy/logZ 별도 optimizer | Adam/AdamW 선택 가능, 두 optimizer | AdamW, parameter group별 LR | Adam, policy/logZ 별도 optimizer | H처럼 AdamW 두 groups, 공통 half-life schedule |
| Gradient clip | 설정별 방식; `norm`은 parameter별 | 같은 방식 | Policy parameter 전체 norm | R 계열 | H처럼 policy 전체 norm, logZ 제외. Threshold도 100으로 정렬 |
| EMA | Sampling model averaging | 같은 계열 | Decay>0일 때 별도 sampling model | 같은 계열 | EMA 수식 공통. 현재 기본 0.99이며 항상 별도 sampling model 보유 |
| Replay | FIFO ring, `choice` 기본 복원 추출 | FIFO ring, 복원 추출 | FIFO deque, 비복원 추출 | FIFO ring, 복원 추출 | FIFO ring list, 비복원 추출은 H 기준. Fresh 삽입 전 old replay 선택 |
| Replay warmup | Generic data pipeline | Generic data pipeline | Configurable warmup, 기본 replay off | Generic data pipeline | 별도 warmup 없음. 가능한 만큼 즉시 replay, 기본 capacity 10000 |
| Checkpoint | Base framework model/config 저장 경로 | Base framework model/config 저장 경로 | Model/config/step/optional EMA 저장 | 별도 agent save에 optimizer/scheduler/replay/CPU·CUDA·NumPy RNG까지 저장 | Model/EMA/optimizer/scheduler/replay/RNG/config/env signature/reward class. Full restart 개념은 X에도 존재하나 형식은 독자적 |
| Public API | Generic GFlowNet trainer/task stack | Generic GFN + CGFlow 통합 | Client/generator/trainer stack | Generic GFN trainer/task stack | 최소 local trainer/sampler/reward/config 및 private gflownet, CLI prepare/train/sample |

현재 backward의 branch weight는 `w(a,parent)=Σ A^(-d)`이고, `log PB = log(w_matching) - log(Σ w_all)`이다. 이것은 새로 만든 임의의 깊이 가중식이 아니라 R에 있는 식을 현재 MDP로 옮긴 것이다. 다만 A의 action count 정의, search pruning, 동일 parent 판별은 달라서 reference와 모든 backward 값이 같지는 않다. Reverse chemistry는 forward로 검증하지만, 모든 precursor의 property budget·남은 길이를 포함한 정확한 MDP 역전이 분포를 계산한 것은 아니다. “Backward는 너무 엄밀하게 가지 않는다”는 합의에 따른 근사다.

현재 TB는 `mean((logZ + ΣlogPF - ΣlogPB - beta×log(max(raw_reward,floor)))²)`다. Failed selected action도 forward 항을 유지하고 raw reward 0을 준다. HSX의 MSE 수식을 따랐다는 말은 이 residual의 형태를 가리키며, rollout distribution·PB·reward floor가 같아서 학습이 동일하다는 뜻은 아니다.

Replay는 “uniform FIFO”라는 이름만으로 네 구현이 같지 않다. H/current는 한 update 안의 replay 추출이 비복원이고, R/C/X의 `choice(len(buffer), batch_size)`는 복원 추출이다. H/current는 새 sample이 같은 update의 replay로 즉시 다시 들어가는 것을 피하지만, 초기 buffer가 작을 때 warmup 및 batch 크기 동작은 다르다.

근거: 현재 [retrosynthesis.py](../../src/rxnflow/envs/retrosynthesis.py):32,248, [trainer.py](../../src/rxnflow/trainer.py):32,93,178,222, [replay.py](../../src/rxnflow/gflownet/replay.py):10. R `src/rxnflow/algo/synthetic_path_sampling.py`:301, C 같은 파일:257, H `src/rxnflow/clients/utils/algo.py`:103, X `src/rxnflow/algo/synthetic_path_sampling.py`:185. H `src/rxnflow/gflownet/algo/trajectory_balance.py`, `gflownet/online_trainer.py`, `gflownet/data/replay_buffer.py`; R/C/X `src/gflownet/online_trainer.py`, `src/gflownet/data/replay_buffer.py`; X `src/rxnflow/base/trainer.py`:181.

## 7. HSX main과 다른 기본값

각 source의 기본 설정 비교다. 개별 실험 스크립트가 override한 실험값과 구분해야 한다.

| 항목 | 현재 | HSX main | 해석 |
| --- | --- | --- | --- |
| Graph hidden / heads / layers | 128 / 2 / 4 | 128 / 2 / 4 | 후속 요청으로 정렬. 전체 architecture까지 같은 것은 아님 |
| Graph readout 차원 | 256=2H | 128, projection 적용 | Policy input 및 action embedding 차원이 다름 |
| Block embedding 차원 | 128 | 128 계열 | 후속 요청으로 정렬. Fusion hidden layer 수는 기존 한 층 유지 |
| Similarity temperature 초기값 | 0.2 | 0.2 | 후속 요청으로 정렬. 범위 0.01–10도 같음 |
| Property mask tolerance | 1% | 1% | 후속 요청으로 정렬. Sampled-row 적용은 유지 |
| Policy LR / logZ LR | 1e-4 / 0.1 | 1e-4 / 0.1 | 같음 |
| LR half-life | 20000 | 20000 | 같음 |
| Weight decay | 1e-8 | 1e-4 | 현재 수치는 R base trainer에서도 사용하지만, 현재 optimizer는 AdamW |
| Policy gradient clip | 100 | 100 | 수식과 threshold 모두 같음. logZ 제외 |
| Reward beta | 32 | 32 | 같음 |
| Reward floor | 1e-4 | 0.01 | 사용자가 1e-4 유지. 낮은/invalid reward에 대한 TB target 차이 |
| Random action probability | 0.1 | 0.2 | 사용자 선택. Exploration 크기와 구현 모두 다름 |
| EMA decay | 0.99 | 0.0 | H default는 별도 EMA를 사용하지 않음 |
| Replay | 기본 on, 64개, capacity 10000 | 기본 off, size/warmup 별도 설정 | FIFO 정책만 같고 기본 학습 분포는 다름 |

WD는 변경 지시가 없어 1e-8을 유지했다. 현재 AdamW policy LR1e-4에서 직접 decay 항은 한 update당 약 1e-12이므로 regularization 효과가 거의 없는 기준이다. Main의 1e-4로 정렬할지는 모델 성능과 parameter norm을 보고 판단할 별도 튜닝 항목이다. Clip100은 10보다 gradient 축소가 덜 개입하는 설정이며, 항상 더 안정적이라는 뜻은 아니다. Reward floor1e-4는 log(0)을 피하고 invalid trajectory에도 유한한 TB target을 준다. Exponent32에서는 최소 log reward가 약 -294.73이므로 낮은 reward에 강한 penalty를 주는 기존 선택을 유지한다.

근거: 현재 [config.py](../../src/rxnflow/config.py), [trainer.py](../../src/rxnflow/trainer.py), [rxnflow.py](../../src/rxnflow/models/rxnflow.py); H `src/rxnflow/config.py`, `models/config.py`, `gflownet/algo/config.py`, `gflownet/data/config.py`; X `src/rxnflow/models/config.py`; R `src/rxnflow/base/trainer.py`.

## 8. 다음 검토에서 구분할 것

### 합의된 차이로 유지하는 부분

- Enamine synthon preparation, 원본 BB mapping, oriented brick/linker, protected type, workflow/Stop 제거, terminal UniReaction은 현재 환경 계약이다.
- Native Torch/fixed graph capacity, dummy isotope를 보존하는 count FP와 uint8, Synple At/+29 보정 제외는 환경 및 의존성 요구사항이다.
- Property normalization scale은 사용자가 간단한 수치로 조정한 설정이며 유지한다. 이를 이식 누락으로 분류했던 설명을 정정한다.
- Uniform per-library sampling, tier/clustering 제외, batch의 state/reaction 간 library draw 공유는 사용자 선택이다. H main의 clustering으로 자동 회귀할 대상이 아니다.
- H의 Linear→LN block projection과 normalized dot/learnable temperature, X의 fusion MLP·readout·post-GNN conditioning, R식 backward를 조합한 방향은 기존 선택이다. 다만 아래 세부 parameter 차이가 모두 별도 합의됐다는 뜻은 아니다.
- Local reward, 공개 범위의 최소 GFlowNet, 서비스/docking/workflow 확장 제외는 프로젝트 범위다.

### 동일하다고 간주하면 안 되는 잔여 차이

| 우선 검토 부분 | 현재 차이 | 왜 별도 검토가 필요한가 |
| --- | --- | --- |
| Atom feature | H의 rich categorical 대신 현재 정의 | Bond stereo는 정렬 완료. Atom feature 전체까지 H와 동일해진 것은 아님 |
| Graph bias/empty state | GENConv bias False vs H True, empty node 처리 다름 | 사용자 요청으로 추후 architecture 작업까지 보류. 현재 수식 test 통과가 H 전체 equivalence의 증거는 아님 |
| Fusion/policy MLP | 현재 Linear→LN→SiLU vs H Linear→SiLU→LN, fusion 한 hidden layer 유지 | 순서 유지 권고. Output/embedding 초기화와 block_dim은 정렬 완료 |
| Training defaults | WD/EMA/replay는 기존 값, reward floor 1e-4와 random 0.1은 사용자 선택 | Graph/block 크기·temperature·clip은 정렬 완료. WD1e-8은 실질적으로 decay가 거의 없는 기준이며 향후 학습 비교로 평가 |
| Mask 적용/failure | Sampled rows에 hard -inf, invalid 종료, H의 fallback 미사용 | Margin은 정렬 완료. Sampling 및 실패 분포까지 같은 것은 아님 |
| Backward approximation | R 기반이지만 search/parent/action count 수정 | H처럼 0도 아니고 R의 bitwise 복제도 아님 |

이 목록은 이번 비교에서 확인한 차이이며, 전부 오류라는 판정이나 성능 개선 제안은 아니다. 특히 어떤 기본값이 더 좋은지는 정적 비교로 결정할 수 없다. 먼저 어떤 HSX 동작을 보존할지 기준을 정한 뒤 해당 부분만 수정해야 한다. 현재 baseline commit을 남긴 이유도 이 비교와 후속 변경을 분리하기 위해서다.

## 9. 재현 및 검증

Reference checkout을 바꾸지 않고 아래처럼 해당 commit의 파일을 읽을 수 있다.

```bash
git show a39c7aebd45fddfdf72109ffa0f8e4b92f9c4458:src/rxnflow/policy/action_categorical.py
git -C source/CGFlow show 89fe021304164764f4c027da2ee0c274de3065fa:src/rxnflow/policy/action_space_subsampling.py
git -C source/rxnflow_hits show e999c3d1b2f911d1b33fb8245c0a66a2946f14b0:src/rxnflow/models/graph_transformer.py
git -C source/rxnflow_hits show 2d472443806ac107b9cdc8a65d03866302394d84:src/rxnflow/models/gfn.py
git show 2aac819:src/rxnflow/models/rxnflow.py
```

최초 phase는 구현 commit 후 정적 비교와 문서 작성만 수행했다. 그 뒤 승인된 세 변경은 별도 구현 phase로 기록한다. Reference 모델 실행, 장기 학습, 새로운 GPU benchmark는 수행하지 않았다. 기존 [복원 기록](implementation-deviations.md)과 [HSX 선택적 이식 기록](hsx-port.md)은 변경 이력이며, 현재의 세부 동일성 판단은 이 문서를 함께 봐야 한다.

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-fourway-comparison-quick.log 2>&1` — exit 0, 52 passed / 1 heavy deselected, 11.69s. Compile/lint/build/import도 통과했다. 이는 현재 코드 검증이며 네 reference와의 동등성 검증이 아니다.
- 최초 비교의 `git diff --check` — 통과. 당시 source와 baseline commit 사이의 차이는 없었다.

후속 구현에서 bond input이 7→13차원으로 바뀌었으므로 이전 checkpoint는 호환되지 않는다. Prepared library의 FP/property 형식은 같아 재생성할 필요가 없다. 사용자 지정 property scale, MW 함수, GENConv bias, encoder/readout과 학습 기본값은 이번 phase에서 변경하지 않았다.

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-hsx-alignment-final-quick.log 2>&1` — exit 0, 53 passed / 1 heavy deselected, 12.29s. Compile/lint/build/import 통과. E/Z/unspecified가 graph input과 GNN output에서 구분되는지, temperature 초기값과 gradient, 1% margin 경계 및 zero/negative bound·strict graph capacity를 확인했다.
- `git diff --check` — 통과. `features.py`와 `config.py`는 변경하지 않았으며 property scale을 유지했다.

추가 승인된 block projection 단순화: FP/property별 Linear→SiLU→Linear→LN→SiLU를 Linear→LN으로 바꾸고 H의 Xavier/zero-bias 초기화를 적용했다. Fusion/policy MLP와 type embedding, readout/conditioning은 유지한다. 이전 two-Linear branch checkpoint는 호환되지 않지만 prepared library는 그대로 사용한다.

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-block-projection-quick.log 2>&1` — exit 0, 53 passed / 1 heavy deselected, 11.46s. Compile/lint/build/import와 기존 graph/block scoring·gradient·train/restart/sample 검증 통과. Runtime 또는 학습 품질 우위를 측정한 것은 아니다.

추가 승인된 model/default phase: graph/fusion/policy MLP output은 Xavier/zero bias, reaction/type embedding은 uniform[-0.1,0.1]. 기본 graph hidden/heads/layers는 128/2/4, block_dim은 128, policy clip은 100, random probability는 0.1로 변경했다. Reward floor1e-4와 WD1e-8, hidden MLP 깊이·activation·normalization 순서·readout·late conditioning은 유지했다. GENConv bias와 empty state는 사용자 요청으로 보류했다. 이전 default 크기의 checkpoint를 새 default 크기 모델에 로드할 수 없으며, 새 baseline은 fresh model로 시작해야 한다. Prepared library는 바뀌지 않았다.

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./test.sh quick > /tmp/rxnflow-model-defaults-quick.log 2>&1` — exit 0, 53 passed / 1 heavy deselected, 11.84s. Compile/lint/build/import와 scoring/gradient/train/restart/sample 검증 통과. 새 default로 GPU runtime 또는 장기 학습은 실행하지 않았다.

### MW CPU 측정

`.venv/bin/python runs/reference_choices_20261002/measure_mw.py`로 local master CPU, Python3.10.12/RDKit2026.03.5에서 측정했다. 기존 seed0 무작위 10,000개 파일의 첫 1,000개 분자(heavy atom 중앙값12)를 미리 파싱하고 warmup한 뒤, 함수별 50회 반복을 7라운드 교대로 실행했다. SMILES parsing, env/model load 및 다른 descriptor는 timing에서 제외했다. 아래 값은 라운드별 분자당 시간의 중앙값이다. Raw JSON과 script는 ignored `runs/reference_choices_20261002/`에 있다.

| 호출 | µs/분자 |
| --- | ---: |
| 변경 전 `Descriptors.MolWt` | 0.939 |
| H `rdMolDescriptors.CalcExactMolWt` | 0.656 |
| Wrapper `Descriptors.ExactMolWt` | 0.910 |

실제 H 호출이 약 0.28µs 빠르지만 wrapper를 사용하는 두 descriptor의 차이는 약 0.03µs다. 이 결과만으로 exact mass 알고리즘이 본질적으로 더 빠르다고 판단하지 않는다. Block property는 prepare 시 계산해 저장하므로 이 차이가 모든 후보 action마다 발생하지도 않는다. [RDKit 문서](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html)에서 MolWt는 average molecular weight, ExactMolWt는 exact molecular weight로 정의된다. 서로 다른 값이므로 성능 최적화 목적으로 교체할 항목으로 보지 않고, 이번에는 현재 MolWt를 유지했다.

후속 사용자 결정: 속도보다 의미가 명확한 `Descriptors.ExactMolWt` API를 선택했다. 공통 `molecular_properties`를 변경하여 state와 새로 준비하는 block 모두 exact mass를 사용한다. 기존 MolWt 기반 NPZ는 다음 사용 전에 `features_stage(env_dir, num_workers=...)`로 다시 생성해야 한다. Synthon conversion 및 fingerprint 정의는 바뀌지 않았다. 이번 phase에서는 production artifact 재생성을 실행하지 않았다.
