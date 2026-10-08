"""Sequential-study client retaining local BatchNorm when explicitly requested."""

from client.client import FedMedClient
from models.cnn import get_parameters


class HeterogeneousClient(FedMedClient):
    def __init__(self, *args, bn_keys=(), local_bn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_bn = local_bn
        self.bn_indices = [i for i, key in enumerate(self.model.state_dict()) if key in bn_keys]
        initial = get_parameters(self.model)
        self.bn_state = {i: initial[i].copy() for i in self.bn_indices}

    def _with_local_bn(self, parameters):
        if not self.local_bn:
            return parameters
        return [self.bn_state[i].copy() if i in self.bn_state else p for i, p in enumerate(parameters)]

    def fit(self, parameters, config):
        result = super().fit(self._with_local_bn(parameters), config)
        if self.local_bn:
            current = get_parameters(self.model)
            self.bn_state = {i: current[i].copy() for i in self.bn_indices}
        return result

    def evaluate(self, parameters, config):
        return super().evaluate(self._with_local_bn(parameters), config)
