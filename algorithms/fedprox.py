"""FedProx uses weighted server averaging and proximal client optimization."""
from algorithms.fedavg import FedMedFedAvg


class FedMedFedProx(FedMedFedAvg):
    """Transmit mu to clients; client.train applies the actual proximal penalty."""

    def fit_config(self, server_round):
        config = super().fit_config(server_round)
        if config["algorithm"] not in ("fedprox", "dynamu"):
            raise ValueError("FedProx requires fedprox or dynamu client optimization")
        return config


def proximal_penalty(model, reference, mu):
    """FedProx objective: mu/2 times squared distance to the round-start model."""
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    parameters = list(model.parameters())
    if len(parameters) != len(reference):
        raise ValueError("Reference parameter count does not match the model")
    penalty = sum((parameter - initial).square().sum()
                  for parameter, initial in zip(parameters, reference))
    return (mu / 2) * penalty
