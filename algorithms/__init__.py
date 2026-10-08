"""Aggregation registry. Extension slots never silently fall back to FedAvg."""

STRATEGIES = {"fedavg": "implemented", "fedprox": "implemented",
              "head_prox": "implemented",
              "coverage_prox": "implemented", "logit_calibration": "implemented",
              "coverage_calibrated": "implemented", "calibrated_fedprox": "implemented",
              "calibrated_uniform_head": "implemented", "relative_coverage_calibrated": "implemented",
              "fednova": "planned", "proposed": "planned"}


def get_strategy(name, *args, **kwargs):
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    if STRATEGIES[name] == "planned":
        raise NotImplementedError(f"{name} is an extension slot, not an implemented algorithm")
    from algorithms.fedavg import FedMedFedAvg
    from algorithms.fedprox import FedMedFedProx
    research_policy = kwargs.pop("research_policy", None)
    strategy = FedMedFedProx if name == "fedprox" else FedMedFedAvg
    if research_policy is not None:
        from algorithms.heterogeneous import HeterogeneousFedAvg
        strategy = HeterogeneousFedAvg
        kwargs["research_policy"] = research_policy
    if len(args) >= 3 and args[2].config["algorithm"]["name"] != name:
        raise ValueError("Strategy name and experiment configuration disagree")
    if "artifacts" in kwargs and kwargs["artifacts"].config["algorithm"]["name"] != name:
        raise ValueError("Strategy name and experiment configuration disagree")
    return strategy(*args, **kwargs)
