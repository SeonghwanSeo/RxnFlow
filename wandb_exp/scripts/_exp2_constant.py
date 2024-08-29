POCKET_DB_PATH = "./data/experiments/sbdd/train_db.pt"

TEST_POCKET_DIR = "./data/experiments/sbdd/protein/test/"
TEST_POCKET_CENTER_INFO: dict[str, tuple[float, float, float]] = {}
with open("./data/experiments/sbdd/center_info/test.csv") as f:
    for line in f.readlines():
        pocket_name, x, y, z = line.split(",")
        TEST_POCKET_CENTER_INFO[pocket_name] = (float(x), float(y), float(z))
