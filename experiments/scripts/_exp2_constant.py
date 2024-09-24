import os

ROOT_DIR = "./data/experiments/CrossDocked2020/"
POCKET_DB_PATH = os.path.join(ROOT_DIR, "train_db.pt")

TEST_POCKET_DIR = os.path.join(ROOT_DIR, "protein/test/")

TEST_POCKET_CENTER_INFO: dict[str, tuple[float, float, float]] = {}
with open(os.path.join(ROOT_DIR, "center_info/test.csv")) as f:
    for line in f.readlines():
        pocket_name, x, y, z = line.split(",")
        TEST_POCKET_CENTER_INFO[pocket_name] = (float(x), float(y), float(z))
