from typing import Dict, List, Union

import torch

from typing import Optional
from enum import IntEnum
from dataclasses import dataclass
from torch import Tensor
from Trajectron.trajec_model.datawrapper.state import StateTensor 

class AgentType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    BICYCLE = 3
    MOTORCYCLE = 4
    
class PadDirection(IntEnum):
    BEFORE = 0
    AFTER = 1
    
@dataclass
class AgentBatch:
    dt: Tensor
    pos_to_vel_fac: float
    agent_type: Tensor
    agent_hist: StateTensor
    agent_hist_len: Tensor
    agent_fut: StateTensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: StateTensor
    neigh_hist_len: Tensor
    robot_fut: Optional[StateTensor]
    robot_fut_len: Optional[Tensor]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "pos_to_vel_fac",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "neigh_hist_len",
            "neigh_types",
            "num_neigh",
            "robot_fut_len",
        }
        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                tensor_val: Union[Tensor, StateTensor]
                setattr(self, val, tensor_val.to(device, non_blocking=True))

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType):
        match_type = self.agent_type == agent_type
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.Tensor):
        """Build a new batch with elements for which filter_mask[i] == True."""

        # Some of the tensors might be on different devices, so we define some convenience functions
        # to make sure the filter_mask is always on the same device as the tensor we are indexing.
        filter_mask_dict = {}
        filter_mask_dict["cpu"] = filter_mask.to("cpu")
        filter_mask_dict[str(self.agent_hist.device)] = filter_mask.to(
            self.agent_hist.device
        )

        _filter = lambda tensor: tensor[filter_mask_dict[str(tensor.device)]]
        _filter_tensor_or_list = lambda tensor_or_list: (
            _filter(tensor_or_list)
            if isinstance(tensor_or_list, torch.Tensor)
            else type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if filter_mask_dict["cpu"][idx]
                ]
            )
        )

        return AgentBatch(
            dt=_filter(self.dt),
            pos_to_vel_fac=self.pos_to_vel_fac,
            agent_type=_filter(self.agent_type),
            agent_hist=_filter(self.agent_hist),
            agent_hist_len=_filter(self.agent_hist_len),
            agent_fut=_filter(self.agent_fut),
            agent_fut_len=_filter(self.agent_fut_len),
            num_neigh=_filter(self.num_neigh),
            neigh_types=_filter(self.neigh_types),
            neigh_hist=_filter(self.neigh_hist),
            neigh_hist_len=_filter(self.neigh_hist_len),
            robot_fut=_filter(self.robot_fut) if self.robot_fut is not None else None,
            robot_fut_len=_filter(self.robot_fut_len)
            if self.robot_fut_len is not None
            else None,
            maps=_filter(self.maps) if self.maps is not None else None,
            maps_resolution=_filter(self.maps_resolution)
            if self.maps_resolution is not None
            else None,
        )