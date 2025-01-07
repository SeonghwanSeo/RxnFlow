from pathlib import Path
from omegaconf import OmegaConf


protocol_config = OmegaConf.load("./template/protocol.yaml")


firstblock_protocols: dict[str, dict] = {}
pattern_to_blocks: dict[int, list[str]] = {}

for block_file in Path("./envs/real/blocks/").iterdir():
    protocol_name = block_key = block_file.stem
    with block_file.open() as f:
        if len(f.readline()) == 0:
            continue
    for pattern_key in block_key.split("-"):
        pattern = int(pattern_key[5:])
        pattern_to_blocks.setdefault(pattern, []).append(block_key)

    # TODO: Remove here
    if "-" in block_key:
        continue

    firstblock_protocols[protocol_name] = {"block_types": [block_key]}

pattern_to_blocks = {k: sorted(list(set(v))) for k, v in pattern_to_blocks.items()}
pattern_to_bricks = {k: [block_key for block_key in v if ("-" not in block_key)] for k, v in pattern_to_blocks.items()}
pattern_to_linkers = {k: [block_key for block_key in v if ("-" in block_key)] for k, v in pattern_to_blocks.items()}

unirxn_protocols: dict[str, dict] = {}

birxn_protocols: dict[str, dict] = {}
for rxn_name, cfg in protocol_config.items():
    rxn_name = str(rxn_name)
    print(cfg)
    if cfg.ordered:
        block_orders = [0, 1]
    else:
        assert cfg.block_type[0] == cfg.block_type[1]
        block_orders = [0]

    for order in block_orders:
        is_block_first = order == 0
        state_pattern = cfg.block_type[1 - order]
        bricks = pattern_to_bricks[cfg.block_type[order]]
        linkers = pattern_to_linkers[cfg.block_type[order]]
        if len(bricks) > 0:
            protocol_name = rxn_name + "_brick_" + ("b0" if is_block_first else "b1")
            protocol_cfg = {
                "forward": cfg.forward,
                "reverse": cfg.reverse,
                "is_block_first": order == 0,
                "state_pattern": state_pattern,
                "block_types": bricks,
            }
            birxn_protocols[protocol_name] = protocol_cfg
        if len(linkers) > 0:
            protocol_name = rxn_name + "_linker_" + ("b0" if is_block_first else "b1")
            protocol_cfg = {
                "forward": cfg.forward,
                "reverse": cfg.reverse,
                "is_block_first": order == 0,
                "state_pattern": state_pattern,
                "block_types": linkers,
            }
            birxn_protocols[protocol_name] = protocol_cfg

OmegaConf.save(
    {"FirstBlock": firstblock_protocols, "UniRxn": unirxn_protocols, "BiRxn": birxn_protocols},
    "./envs/real/protocol.yaml",
)
