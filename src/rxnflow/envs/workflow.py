from collections.abc import Iterable
from pathlib import Path
from omegaconf import ListConfig, OmegaConf
from .action import RxnActionType, Protocol


class BaseWorkflowNode:
    def __init__(self, name, parent):
        self.name: str = name
        self.parent: BaseWorkflowNode | None = parent
        self.children: list[WorkflowNode] = []

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, node):
        self.children.append(node)

    def valid_check(self) -> bool:
        if self.is_leaf:
            return True
        flag = True
        assert len(self.children) == len(set(node.name for node in self.children))
        for node in self.children:
            flag = flag and node.valid_check()
        return flag


class WorkflowNode(BaseWorkflowNode):
    def __init__(self, name: str, protocol: Protocol, parent: BaseWorkflowNode):
        super().__init__(name, parent)
        assert self.parent is not None
        self.protocol: Protocol = protocol
        self.parent.add_child(self)

    def iteration(self, prev_traj: list | None = None) -> Iterable[list[Protocol]]:
        if prev_traj is None:
            prev_traj = []
        if self.is_leaf:
            yield prev_traj + [self.protocol]
        else:
            for node in self.children:
                yield from node.iteration(prev_traj + [self.protocol])


class WorkflowRoot(BaseWorkflowNode):
    def __init__(self):
        super().__init__("root", None)

    def iteration(self) -> Iterable[list[Protocol]]:
        for node in self.children:
            yield from node.iteration()


def read_config(
    node: BaseWorkflowNode,
    child_config_list: str | list[str | dict] | ListConfig,
    protocol_dict: dict[str, Protocol],
):
    for child_cfg in child_config_list:
        if isinstance(child_cfg, str):
            name = str(child_cfg)
            WorkflowNode(name, protocol_dict[name], node)
        else:
            assert len(child_cfg) == 1
            name, child_child_config_list = next(iter(child_cfg.items()))
            child_node = WorkflowNode(name, protocol_dict[name], node)
            read_config(child_node, child_child_config_list, protocol_dict)


class Workflow:
    def __init__(self, workflow_config_path: str | Path, protocol_config_path: str | Path):
        workflow_config = OmegaConf.load(workflow_config_path)
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
                protocol = Protocol(name, action, **cfg)
                self.protocols.append(protocol)
        self.protocol_dict = {protocol.name: protocol for protocol in self.protocols}

        self.root = WorkflowRoot()
        read_config(self.root, workflow_config, self.protocol_dict)
