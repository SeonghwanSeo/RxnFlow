# Project layout

`src/rxnflow` is the current implementation. Its subpackages follow responsibilities, and the package root contains the small public training, sampling, configuration, and reward APIs.

```text
src/rxnflow/
├── envs/
│   ├── chemistry/
│   │   ├── features.py     # RDKit parsing, descriptors, fingerprints (NumPy)
│   │   ├── synthon.py      # Synthon definitions and conversions
│   │   └── reaction.py     # Forward/reverse UniReaction and BiReaction
│   ├── prepare.py          # Enamine input → libraries, provenance, NPZ
│   ├── library.py          # Prepared library loading and aligned tensors
│   ├── graph.py            # Synthon graph features, fixed tensors, batching
│   ├── env.py              # State transitions, feasible actions, termination
│   └── retrosynthesis.py   # Bounded reverse routes and backward probabilities
├── models/
│   ├── graph_transformer.py # Neural graph encoder
│   └── rxnflow.py           # Reaction and block scoring
├── gflownet/
│   ├── policy.py            # SynthesisPolicy: action probabilities and trajectories
│   ├── types.py             # States, actions, trajectories, molecular samples
│   ├── categorical.py       # Masked protocol logits, denominator, device sampling
│   ├── subsampling.py       # Uniform per-library subsampling
│   └── replay.py            # Trajectory replay
├── cli/                    # prepare, train, sample entry points
├── config.py               # Resolved configuration
├── reward.py               # RewardFunction and local evaluation
├── trainer.py              # Training, EMA, optimizer, checkpoints
└── sampler.py              # Checkpoint-backed sampling and result output
```

`envs/` owns the current Enamine synthon environment, including its chemistry and catalog preparation. This groups code by the environment it belongs to while retaining separate modules for distinct responsibilities. The structure targets the one supported synthon environment; it does not introduce a generic multi-environment or workflow interface.

Within `envs/`, both `prepare.py` and `env.py` use `chemistry/`. The chemistry modules do not import preparation or state-transition code, and `env.py` does not import `prepare.py`. `envs/graph.py` derives fixed-size graph tensors and batches from the state's RDKit molecule at the policy input boundary. `models/` contains the neural encoder and action scoring modules. `gflownet/` holds execution, probability estimation, and replay shared by trainer and sampler.

Import the environment directly with `from rxnflow.envs.env import SynthesisEnv`. Preparation is available through `from rxnflow.envs.prepare import convert_stage, features_stage`. There are no compatibility modules at the former `rxnflow.data` or `rxnflow.chemistry` paths. Public entry points such as `from rxnflow.trainer import RxnFlowTrainer` and the CLI commands remain the normal user API.

The root `__init__.py` contains only the package version. Import configuration, rewards, trainers, and samplers from their defining modules. State, action and trajectory dataclasses live in `gflownet/types.py`; reward functions receive RDKit `Mol` objects directly. `MoleculeState` retains an RDKit `Mol` and caches its canonical SMILES; reactions create new molecules while existing state molecules are treated as read-only. Both reaction directions use RDKit `ChemicalReaction`; there is no separate graph-edit engine. `SynthesisPolicy` in `gflownet/policy.py` computes action logits and forward probabilities and samples trajectories; `BlockSubsampler` draws the per-library subsets. `RetrosynthesisSearch` enumerates reverse routes, `RetrosynthesisTree` stores them, and `RetrosynthesisWorkers` distributes that search across processes.

`BiReaction` owns reactant direction and the fixed incoming attachment marker. There is no separate `BiAction` wrapper. Library keys are ordered: `A-B` means attach the isotope-0 dummy as type A and retain the type-B dummy. Symmetric linker orientations share one canonical row.

## Repository-level directories

| Path | Role |
| --- | --- |
| `src/rxnflow/` | Current package implementation |
| `tests/` | Synthetic automated checks and fixture inputs |
| `configs/` | User-facing configuration examples |
| `data/templates/` | Current synthon/reaction YAML and curation references |
| `data/building_blocks/` | Local vendor inputs, excluded from Git |
| `codex/docs/` | Design contracts and implementation review guides |
| `examples/` | Small public API examples |
| `source/` | Local reference implementations: hsx and CGFlow |
| `image/` | Existing project image asset |
| `build/`, `dist/`, `*.egg-info/` | Generated package artifacts |

The source reorganization does not require relocating local vendor data, reference checkouts, or run outputs. Those paths are independent of the Python module responsibilities. `PLAN.md` and `PROGRESS.md` remain the local review checklist and work log.
