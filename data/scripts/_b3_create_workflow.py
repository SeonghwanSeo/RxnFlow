from pathlib import Path
from omegaconf import OmegaConf
import argparse


def get_workflow(env_dir: Path, protocol_dir: Path):
    protocol_config = OmegaConf.load(protocol_dir / "protocol.yaml")
    save_workflow_path = env_dir / "workflow.yaml"

    firstblock_protocols: dict[str, dict] = {}
    unirxn_protocols: dict[str, dict] = {}
    birxn_protocols: dict[str, dict] = {}
    workflow_config = {"FirstBlock": firstblock_protocols, "UniRxn": unirxn_protocols, "BiRxn": birxn_protocols}

    # firstblock
    pattern_to_types: dict[int, list[str]] = {}
    for block_file in Path(env_dir / "blocks/").iterdir():
        block_type = block_file.stem
        protocol_name = "block" + block_type
        with block_file.open() as f:
            if len(f.readline()) == 0:
                continue
        for pattern in map(int, block_type.split("-")):
            pattern_to_types.setdefault(pattern, []).append(block_type)
        # TODO: Remove here, currently, only brick for firstblock
        if "-" in block_type:
            continue
        firstblock_protocols[protocol_name] = {"block_types": [block_type]}

    # remove redundant items
    pattern_to_types = {k: sorted(list(set(v))) for k, v in pattern_to_types.items()}
    pattern_dict = {
        pattern: {
            "brick": [t for t in block_types if ("-" not in t)],
            "linker": [t for t in block_types if ("-" in t)],
        }
        for pattern, block_types in pattern_to_types.items()
    }

    # birxn (no unirxn)
    for rxn_name, cfg in protocol_config.items():
        rxn_name = str(rxn_name)
        if cfg.ordered:
            block_orders = [0, 1]
        else:
            assert cfg.block_type[0] == cfg.block_type[1]
            block_orders = [0]

        for order in block_orders:
            is_block_first = order == 0
            state_pattern = cfg.block_type[1 - order]
            block_pattern = cfg.block_type[order]
            for t in ["brick", "linker"]:
                protocol_name = rxn_name + f"_{t}_" + ("b0" if is_block_first else "b1")
                block_keys = pattern_dict[block_pattern][t]
                if len(block_keys) > 0:
                    birxn_protocols[protocol_name] = {
                        "forward": cfg.forward,
                        "reverse": cfg.reverse,
                        "is_block_first": order == 0,
                        "state_pattern": state_pattern,
                        "block_types": block_keys,
                    }
    OmegaConf.save(workflow_config, save_workflow_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Workflow")
    parser.add_argument(
        "-d",
        "--env_dir",
        type=str,
        help="Path of environment directory",
        default="./envs/real/",
    )
    parser.add_argument(
        "-p",
        "--protocol_dir",
        type=Path,
        help="Path of synthesis protocol",
        default="./template/real/",
    )
    args = parser.parse_args()

    get_workflow(args.env_dir, args.protocol_dir)
