from gflownet.base.base_trainer import BaseTrainer
from gflownet.tasks.unidock_moo_frag import UniDockMOOFragTrainer
from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer
from gflownet.misc.synflownet.unidock_trainer import UniDockMOOSynFlowNetTrainer
from gflownet.misc.rgfn.unidock_trainer import UniDockMOORGFNTrainer

TARGET_DIR = "./data/experiments/LIT-PCBA"
TARGET_CENTER = {
    "ADRB2": (-1.96, -12.27, -48.98),
    "ALDH1": (34.43, -16.88, 13.77),
    "ESR_ago": (-35.22, 4.64, 20.78),
    "ESR_antago": (17.85, 35.51, 52.49),
    "FEN1": (-16.81, -4.80, 0.62),
    "GBA": (32.44, 33.88, -19.56),
    "IDH1": (12.11, 28.09, 80.47),
    "KAT2A": (-0.11, 5.73, 10.14),
    "MAPK1": (-15.69, 14.49, 42.72),
    "MTORC1": (35.38, 49.65, 36.21),
    "OPRK1": (58.61, -24.16, -4.32),
    "PKM2": (8.64, 2.94, 10.76),
    "PPARG": (8.30, -1.02, 46.32),
    "TP53": (89.32, 91.82, -44.87),
    "VDR": (11.38, -3.12, -31.57),
}

TRAINER_DICT: dict[str, type[BaseTrainer]] = {
    "frag": UniDockMOOFragTrainer,
    "rxnflow": UniDockMOOSynthesisTrainer,
    "synflownet": UniDockMOOSynFlowNetTrainer,
    "rgfn": UniDockMOORGFNTrainer,
}
