from pathlib import Path
import argparse

from _b1_smi_to_block import get_block
from _b2_save_block_data import get_block_data
from _b3_create_workflow import get_workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create environment")
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=Path,
        help="Path of input building block smiles file",
        default="./building_blocks/enamine_stock.smi",
    )
    parser.add_argument(
        "-p",
        "--protocol_dir",
        type=Path,
        help="Path of input synthesis protocol directory",
        default="./template/real/",
    )
    parser.add_argument(
        "-o",
        "--env_dir",
        type=Path,
        help="Path of output environment directory",
        default="./envs/stock/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers", default=1)
    args = parser.parse_args()

    env_dir: Path = args.env_dir
    protocol_dir: Path = args.protocol_dir
    block_file: Path = args.building_block_path
    num_cpus: int = args.cpu

    assert not env_dir.exists()

    print("convert building blocks to ready-to-compose fragments")
    get_block(env_dir, block_file, protocol_dir, num_cpus)
    print("pre-calculate building block features")
    get_block_data(env_dir, num_cpus)
    print("create workflow")
    get_workflow(env_dir, protocol_dir)
