import argparse

from _b_smi_to_env import get_block_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=str,
        help="Path to input enamine building block file (.smi | .smi.gz)",
        default="./building_blocks/enamine_catalog.smi",
    )
    parser.add_argument(
        "-t",
        "--template_path",
        type=str,
        help="Path to reaction template file",
        default="./templates/real.txt",
    )
    parser.add_argument(
        "-o",
        "--save_directory",
        type=str,
        help="Path to environment directory",
        default="./envs/catalog/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    get_block_data(args.building_block_path, args.template_path, args.save_directory, args.cpu)
