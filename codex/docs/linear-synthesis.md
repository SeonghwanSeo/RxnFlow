# 선형 synthon 합성 구현 검토 안내

## 검토 순서

폴더별 책임과 의존 방향은 [프로젝트 구조](project-layout.md)를 기준으로 한다.

1. `data/templates/synthon.yaml`, `reaction.yaml`, `src/rxnflow/envs/chemistry/synthon.py`, `envs/prepare.py`: synthon 의미, 변환 범위, brick/linker 생성과 원본 BB mapping.
2. `src/rxnflow/envs/chemistry/reaction.py`, `envs/env.py`: 반응 결과 열거, action identity, 마지막 step과 property masking.
3. `src/rxnflow/gflownet/types.py`, `gflownet/policy.py`, `envs/retrosynthesis.py`: 상태 전이, 종료, invalid 처리, forward/backward 확률.
4. `src/rxnflow/models/rxnflow.py`, `envs/graph.py`: 입력 feature와 reaction/block scoring.
5. `src/rxnflow/trainer.py`, `sampler.py`: reward, replay·EMA, checkpoint, 경로 출력.

## 상태와 종료

상태는 현재 synthon의 RDKit `Mol`, 반응 횟수, 종료 여부다. Canonical synthon SMILES는 식별·출력·역합성 검색을 위해 캐시한다. 여기서 Mol은 dummy를 포함한 현재 synthon이며, 원본 BB나 복원된 실제 중간체를 병행 보존하는 뜻은 아니다. State의 Mol은 읽기 전용으로 다루고, 반응은 새로운 Mol을 반환한다. Forward와 reverse 모두 기존 YAML의 SMARTS를 RDKit `ChemicalReaction`으로 실행한다. State의 dummy isotope는 site의 종류이며, 보호된 site도 하나로 센다. Catalog에서는 결합할 dummy만 isotope 0 (`[*]`)으로 고정하고, 그 type은 library key의 첫 값에 저장한다. `A-B` linker의 반대편 dummy는 type B를 유지한다. Brick도 같은 marker를 쓰며 FirstBlock에서 type을 복원한다. Catalog에는 site 1개인 brick과 2개인 linker만 존재하고, 생성 중인 중간체에는 항상 site가 1개 있다. 동일 원본에서 만들어진 서로 다른 synthon은 별도 상태/블록이다.

| Action | 입력 | 출력 | 반응 횟수 |
| --- | --- | --- | --- |
| FirstBlock | 빈 상태 + brick | site 1개 | 0 |
| BiReaction + linker | site 1개 + site 2개 | site 1개 | +1 |
| BiReaction + brick | site 1개 + site 1개 | site 0개, 즉시 종료 | +1 |
| UniReaction | site 1개 | site 1개 또는 0개 | +1 |

Stop, restore, terminal BB 선택은 없다. `min_reactions` 전에는 종료형 action을 가리고, `max_reactions`의 마지막 반응에서는 종료형 action만 허용한다. 종료한 분자는 다시 활성화하지 않는다. 기본 `max_reactions=3`은 결합 → 탈보호 → 재결합을 포함할 수 있다. 아무 후보도 남지 않으면 invalid trajectory로 처리한다.

Linker의 결합 방향은 준비 단계에서 별도 block row로 저장하므로, block 선택으로 위치 선택이 끝난다. 같은 두 type이라도 비대칭 위치는 두 row이고, 대칭으로 동일해지는 방향은 하나로 합친다. 고정된 reaction/block 선택에서는 연결된 생성물이 하나여야 한다. 대칭 중복 제거 후 서로 다른 생성물이 여러 개면 template ambiguity 오류로 처리한다. Action은 반응 이름, block type/index, `product_smiles`로 선택을 기록한다. 대칭 위치가 동일 생성물로 귀결되면 한 action으로 합친다. Atom index는 canonicalization에 따라 바뀌므로 영구 site ID로 사용하지 않는다. 전체 분자의 새 작용기를 자동 재인식하는 단계는 없다.

## Synthon type 목록

아래는 현재 SMARTS의 의미를 요약한 것이며, 실제 매칭 제약은 YAML이 기준이다. Marker가 남아 있는 구조는 추상 표현이므로, 원본 작용기의 원자가 일부 생략되거나 고리 구조가 미리 표현될 수 있다. 기존 27–32도 이 원칙을 따른다.

| Type | 인식하는 전구체/작용기 | 표현의 핵심 |
| --- | --- | --- |
| 1 | amine | N 유지, `N-[1*]` |
| 2 | OH 또는 N-H | O/N 유지, heteroatom에 `[2*]` 부착 |
| 3 | carboxylic acid | 전체 `C(=O)O`를 `[3*]`로 치환 |
| 4 | aldehyde | `CHO`를 `[4*]`로 치환 |
| 5 | acyl halide, 지정 ester | acyl handle을 `[5*]`로 치환 |
| 6 | acyl hydrazide | `C(=O)NN`을 `[6*]`로 치환 |
| 7 | ketone | `C=O`의 O를 `[7*]`로 치환 |
| 8 | carbon-bound Cl/Br/I | halogen을 `[8*]`로 치환 |
| 9 | substituted hydrazine | `NN`을 `[9*]`로 치환 |
| 10 | azide | azide를 `[10*]`로 치환 |
| 11 | nitrile | `C#N`을 `[11*]`로 치환 |
| 12 | terminal alkyne | `C#CH`를 `[12*]`로 치환 |
| 13 | sulfonyl halide | sulfonyl handle을 `[13*]`로 치환 |
| 14 | primary aliphatic amine | N을 생략한 `C-[14*]` |
| 15 | aliphatic alcohol | O를 생략한 `C-[15*]` |
| 16 | aliphatic thiol | S를 생략한 `C-[16*]` |
| 17 | aliphatic halide | `C-[17*]` |
| 18 | aliphatic chloride / 지정 sulfonate | `C-[18*]` |
| 19 | CH-bound halide | `C-[19*]` |
| 20 | epoxide | 열린 `[*]-C-C-OH` 표현 |
| 21 | aryl halide | `c-[21*]` |
| 22 | phenol | O를 생략한 `c-[22*]` |
| 23 | aryl boronic acid | boronic handle을 `[23*]`로 치환 |
| 24 | 지정 electron-withdrawing group 인접 carbon | `C=[24*]` |
| 25 | 두 electron-withdrawing group 사이 carbon | `C=[25*]` |
| 26 | activated alkene | 단일결합으로 바꾸고 `[26*]` 부착 |
| 27 | amino ester | cyclic dicarbonyl synthon |
| 28 | ortho amino amide | cyclic synthon의 `C=[28*]` |
| 29 | beta-amino acid | cyclic dicarbonyl synthon |
| 30 | ortho amino + N/O/S nucleophile | fused heterocycle 표현 |
| 31 | ortho amino methyl ester | cyclic synthon의 `[31*]` |
| 32 | ortho amino methyl ester | 다른 cyclic synthon 표현의 `[32*]` |
| 33 | N-Boc amine | N 유지, Boc를 `[33*]`로 치환 |
| 34 | methyl carboxylate | 전체 ester를 `[34*]`로 치환 |
| 35 | ethyl carboxylate | 전체 ester를 `[35*]`로 치환 |

현재 UniReaction은 `33→1`(Boc 제거), `34→3`(methyl ester hydrolysis), `35→3`(ethyl ester hydrolysis), `11→terminal tetrazole`이다. 전구체의 보호기 세부 구조는 catalog 변환 단계에서 확인하고, runtime은 typed dummy를 신뢰한다. Type 33–35에는 직접 결합하는 BiReaction이 없으므로 먼저 해당 unary 전환이 필요하다. 같은 ester가 type 5와 34/35에 모두 대응하는 것은 서로 다른 합성 표현으로 보존한다.

## Library와 provenance

입력은 Enamine `SMILES<TAB>ID`다. 염을 제거하고 canonicalize한 원본 구조를 `building_blocks.json`에 기록한다. 각 `blocks/<type>.smi`는 canonical synthon과 JSON ID 목록을 저장한다. Feature artifact는 `bb_feature.npz`이며 `<type>/smiles`, `<type>/properties`, `<type>/fingerprints`, `<type>/heavy_atoms` 배열로 저장한다. Chemistry/preparation은 NumPy를 사용하고 모델 입력 경계에서 PyTorch tensor로 변환한다. CLI는 새 출력 디렉터리에 conversion과 features를 순서대로 실행한다. Resume/force/prepare_all은 제공하지 않는다.

One-site 변환을 먼저 열거한 뒤 각 brick에 모든 변환을 한 번 더 적용해 linker를 만든다. 두 변환의 순서를 모두 시도하며 결과의 dummy signature가 정확히 해당 두 type일 때만 보존한다. 동일 type pair도 포함한다. 이미 marking된 amine에 다시 dummy를 붙이는 중복 변환은 허용하지 않는다. 각 linker에서 두 dummy를 번갈아 isotope 0으로 바꿔 `A-B`와 `B-A` library에 저장한다. 같은 type이면 같은 library의 방향별 row가 된다. 이 marker까지 포함한 canonical synthon이 같으면 source ID를 합치고, 다르면 별도 block으로 유지한다.

Provenance는 해당 synthon을 만들 수 있는 원본 BB 후보의 목록이다. 원본 공급자 ID 중 하나를 추가 action으로 선택하지 않는다. Public trajectory에는 `block_ids`, `block_smiles`, `building_blocks`와 선택된 생성물이 포함되므로, 결과에서 source 구조까지 추적할 수 있다.

## Masking과 모델

hsx의 state budget + block budget 방식을 사용한다. 가능한 reaction/library type을 결정하고, 해당 library에서 uniform subsampling한 뒤 선택된 row의 준비된 property와 현재 state property를 합산해 `property_penalty` 및 heavy-atom capacity mask를 계산한다. 확률 보정은 원래 library 크기와 최초 sampling 개수를 기준으로 한다. 탈락 후보를 다시 채우지 않으며, 모든 후보가 탈락하면 trajectory가 실패한다. 0 또는 음수 상한도 지원하도록 bound로 나누지 않고 원래 단위로 비교한다. 0이 아닌 상한은 hsx main의 1% 상대 여유를 적용하고, 0 상한과 heavy-atom capacity는 strict하게 유지한다. Property 합산은 추정치이며 실제 생성물의 상한을 보장하지 않는다. Enamine dummy isotope는 질량에서 제외하며, hsx의 Synple/eXplore 전용 At 질량 차감 및 linker MW +29.0은 이식하지 않는다.

후보별 반응 실행 없이 점수를 계산하고 action을 선택한다. 선택한 action만 RDKit으로 실행하여 site signature와 실제 `max_atoms`를 확인한다. 실패하면 해당 trajectory는 invalid이며 실패 action도 빈 product SMILES로 trajectory에 남겨 TB 학습에 포함한다. 재선택하거나 mask를 풀지 않는다. UniReaction은 typed handle과 step 규칙으로 선택하며 block budget이나 생성물 property 검사는 적용하지 않는다. 선택된 생성물 Mol을 state와 graph encoding에서 재사용한다. Product SMILES는 선택 후 action 기록에 채우며, policy choice의 identity는 reaction과 oriented block row이다.

MW는 `Descriptors.ExactMolWt`로 통일한다. 기존 MolWt 기반 prepared 환경은 사용 전에 features stage를 다시 실행해야 한다. Descriptor는 현재 synthon에 대해 계산하며 dummy isotope가 가짜 원자 질량으로 포함되지 않도록 한다. 이는 복원된 실제 중간체의 물성을 보장한다는 뜻은 아니다. 최종 생성물에는 dummy가 없으므로 물성 및 reward는 그 실제 최종 구조를 평가한다.

`property_penalty`의 상한은 유한한 수를 받는다. `rings: 0`, `hbd: 0`처럼 특정 count를 금지하거나 음수 logP 상한을 지정할 수 있다. 제한하지 않을 property는 mapping에서 생략한다.

Graph는 `max_atoms + 1`개의 고정 node slot을 사용한다. RDKit heavy atom에는 dummy가 포함되지 않으므로 마지막 slot을 예약한다. Atom/type one-hot, degree, charge, aromaticity, mass, H count, hybridization, chirality(CW/CCW/unspecified)를 사용하며 bond feature는 type 4개·stereo 7개·conjugation/ring 2개로 총 13개다. 전역 입력은 property 9개, 남은 heavy-atom capacity, 반응 횟수다. Block은 Morgan count 512 + MACCS 166, property 9개와 library type embedding을 사용한다. Fingerprint는 count를 255에서 포화시킨 뒤 MACCS와 함께 uint8로 저장하고 CPU library에서도 그대로 유지한다. 모델에서 선택된 행만 float32로 변환한다. 기본 Morgan이 dummy isotope를 구분하지 않으므로 typed dummy의 atom invariant를 명시적으로 지정한다. 현재 state의 graph embedding과 reaction embedding으로 UniReaction 점수를 계산하고, FirstBlock/BiReaction에서는 준비된 block fingerprint/property와 library type embedding을 추가로 사용한다. 생성물 fingerprint encoder와 추가 scoring은 사용하지 않는다. 현재 state와 준비된 block의 property 합을 budget masking에 사용한다.

## 학습과 검증 범위

Backward는 생성된 parent branch를 보존하고, 각 reverse rule의 중복 제거된 canonical precursor set을 모두 탐색한다. 후보는 catalog 존재 여부와 forward 재실행으로 확인한다. 반응 횟수 상한까지만 탐색하며, 짧은 경로를 먼저 찾았다는 이유로 다른 경로를 제거하지 않는다. 동일 action이 서로 다른 parent에서 나온 경우 backward 확률을 구분한다. 깊이 가중 확률은 실용적인 근사로 유지하며, 모든 역경로에 대해 state별 property budget/반응 횟수의 정확한 MDP 역전이를 증명하는 구현은 아니다.

TB/replay는 관측 action과 독립적으로 library를 uniform sampling해 분모를 추정한다. Weight는 `log(library_size / sampled_count)`이며 관측 action은 별도로 scoring한다. RxnFlow master처럼 `logP = min(observed_logit - logZ, 0)`를 사용한다. 방향이 고정된 block row 하나가 sampling 단위다. Sampling temperature·importance temperature는 생성 정책에 적용하고 random exploration은 CGFlow의 `-log(n_libraries * n_sampled)` 가중치와 동일한 mask를 사용한다.

Invalid trajectory는 raw reward 0으로 기록하고 학습에서 reward floor를 적용한다. 화학적으로 후보가 없는 경우만 이 경로로 처리하며, 예상하지 못한 모델/runtime 오류는 감추지 않는다. 종료 분자만 reward와 public samples로 반환한다.

Synthetic quick 검증은 선형 경로의 전이·mask·site 선택·backward·학습 연결을 확인한다. Production template의 실험적 적용 범위, 전체 Enamine catalog의 성능 및 분포 품질은 별도 검토 대상이다. hsx 방식 multi-step workflow library는 이번 구현에 포함하지 않는다.

TB loss는 MSE이며, logZ는 policy와 별도 learning rate를 사용한다. Replay는 fresh trajectory를 넣기 전에 기존 buffer에서 균일 비복원 추출한다. Learning rate는 지정한 half-life에 따라 감소하고, optimizer/scheduler 상태를 함께 복원한다. 모델은 FP/property별 Linear→LayerNorm projection(main의 Xavier 초기화), type embedding과 결합하는 fusion MLP, GNN 이후 additive reaction conditioning, block 정규화 dot score 및 학습 temperature를 사용한다. Temperature 범위는 0.01–10이며 초기값은 main과 같은 0.2다. UniReaction scalar score도 같은 temperature convention을 따른다. Graph encoder는 native Torch의 residual GINE, graph-mode normalization과 conditional scale/shift를 사용한다. 각 layer는 ReLU(source+bond)의 이웃 합과 자기 node를 더한 뒤 H→2H→H MLP로 갱신한다. Epsilon은 0으로 고정하며 별도 self-loop와 attention은 없다. Readout은 molecular mean과 virtual node를 concat한 2H에 LayerNorm을 적용한다. 상세 출처와 차이는 [HSX 이식 기록](hsx-port.md)을 참고한다.

기본 모델은 hidden128/layers4, block_dim128이다. GINE 전환으로 num_heads 설정은 삭제했다. Graph/fusion/policy MLP는 hidden Kaiming·output Xavier와 zero bias를 사용하고, reaction/type embedding은 uniform[-0.1,0.1]로 초기화한다. Fusion/policy의 Linear→LN→SiLU 순서와 기존 hidden 깊이, 2H readout, GNN 이후 reaction conditioning은 유지한다. GENConv는 GINE로 대체되어 bias 비교 항목도 사라졌다. Empty state는 기존 virtual node 방식이며 anchor readout은 추가하지 않았다. 학습은 policy 전체 gradient norm100으로 clip하며 logZ는 제외한다. Random probability는 0.1, reward floor는 1e-4, weight decay는 기존 1e-8이다.

Checkpoint에는 library subsampling용 CPU generator와 Gumbel sampling·dropout용 전역 CPU·CUDA RNG 상태를 함께 보관한다. 복원 시 checkpoint를 CPU로 읽고 model/optimizer loader가 parameter를 해당 device로 옮긴다. RNG 상태가 없는 이전 checkpoint의 호환 복원은 제공하지 않는다.

State graph에는 atom chirality와 main의 bond stereo categorical을 사용한다. Bond는 type 4개, stereo 7개(NONE/ANY/Z/E/CIS/TRANS/unknown), conjugation/ring 2개로 총 13차원이며 E/Z를 구분한다. Block fingerprint는 기존 achiral 설정이므로 state stereo 표현이 block encoder의 모든 stereoisomer 구분을 보장하지는 않는다. 유한한 fingerprint와 property가 같은 서로 다른 block에는 같은 embedding을 부여한다. Property normalization scale은 사용자가 조정한 현재 값을 유지한다.

Library subsampling은 policy batch마다 library별로 한 번만 수행하고 모든 state와 reaction이 공유한다. Budget mask는 state별로 broadcast 연산한다. 관측 action의 존재 여부는 sampling에 영향을 주지 않는다. 다음 policy 호출에서는 새로 sampling하며, full set을 쓰는 작은 library는 index tensor를 그대로 재사용하고 RNG를 소비하지 않는다.

Reaction마다 compatible library를 concat하여 `[state, sampled block]` score matrix를 계산한다. Mask된 column은 `-inf`로 남기며 valid-index 압축은 하지 않는다. Device에서 Gumbel sampling한 뒤 선택된 좌표만 CPU로 옮겨 RxnAction으로 변환한다. TB/replay는 관측 action의 logit을 별도로 계산한다. Reverse worker는 다음 forward 계산과 겹쳐 실행하고, 다음 parent tree를 사용하기 전과 마지막 iteration 이후 결과를 수거한다.
