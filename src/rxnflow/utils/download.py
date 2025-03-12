import os

import gdown

DRIVE_ID = {
    # up-to-date version
    "qed-unif-0-64": "1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi",
    "qvina-unif-0-64": "13ATBmJIgIhU3Kclwut0yssjaciPCieEV",
    # all version
    "qed-unif-0-64_20250512": "1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi",
    "qvina-unif-0-64_20250512": "13ATBmJIgIhU3Kclwut0yssjaciPCieEV",
}

FILE_NAME = {
    "1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi": "qed-unif-0-64_20250512.pt",
    "13ATBmJIgIhU3Kclwut0yssjaciPCieEV": "qvina-unif-0-64_20250512.pt",
}


def download_pretrained_weight(pretrain_model: str) -> str:
    if os.path.exists(pretrain_model):
        return pretrain_model
    else:
        if pretrain_model in DRIVE_ID.keys():
            id = DRIVE_ID[pretrain_model]
        elif pretrain_model in FILE_NAME.keys():
            id = pretrain_model
        else:
            raise ValueError(f"{pretrain_model} is not available: it should be model name, gdrive id, or file name.")
        file_name = os.path.join("./weights/", FILE_NAME[id])
        if not os.path.exists(file_name):
            gdown.download(id=id, output=file_name)
        return file_name
