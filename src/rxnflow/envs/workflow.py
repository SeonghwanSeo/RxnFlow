from pathlib import Path
from omegaconf import OmegaConf
from .action import RxnActionType, Protocol
from .reaction import BiRxnReaction, BckBiRxnReaction


class Workflow:
    def __init__(self, protocol_config_path: str | Path):
        protocol_config = OmegaConf.load(protocol_config_path)

        self.protocols: list[Protocol] = [Protocol("stop", RxnActionType.Stop)]
        for action_type, cfg_dict in protocol_config.items():
            action_type = str(action_type)
            if action_type == "FirstBlock":
                action = RxnActionType.FirstBlock
            elif action_type == "UniRxn":
                action = RxnActionType.UniRxn
            elif action_type == "BiRxn":
                action = RxnActionType.BiRxn
            else:
                raise ValueError(action_type)

            for name, cfg in cfg_dict.items():
                if action is RxnActionType.FirstBlock:
                    protocol = Protocol(name, action)
                elif action is RxnActionType.UniRxn:
                    forward = BiRxnReaction(cfg.forward, cfg.is_block_first)
                    reverse = BckBiRxnReaction(cfg.reverse, cfg.is_block_first)
                    protocol = Protocol(name, action, forward=forward, reverse=reverse, block_type=cfg.block_type)
                elif action is RxnActionType.BiRxn:
                    forward = BiRxnReaction(cfg.forward, cfg.is_block_first)
                    reverse = BckBiRxnReaction(cfg.reverse, cfg.is_block_first)
                    protocol = Protocol(name, action, forward=forward, reverse=reverse, block_type=cfg.block_type)
                else:
                    raise ValueError(action_type)
                self.protocols.append(protocol)
        self.protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in self.protocols}
