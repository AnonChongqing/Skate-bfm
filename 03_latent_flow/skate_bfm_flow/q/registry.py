from .aggregators import aggregate
from .losses import critic_loss
from .targets import finite_horizon_target, td_target

AGGREGATORS = {name: name for name in ("min", "mean", "mean_minus_std", "min_minus_disagreement")}
TARGETS = {
    "finite_horizon_return": finite_horizon_target,
    "semi_mdp_td": td_target,
    "sac_td": td_target,
    "td3_td": td_target,
}
LOSSES = {name: name for name in ("huber", "mse", "mae")}

__all__ = ["AGGREGATORS", "TARGETS", "LOSSES", "aggregate", "critic_loss"]
