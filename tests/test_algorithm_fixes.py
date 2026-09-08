import math

import torch

from ConMedRL.conmedrl import FQE, RLTraining
from ConMedRL.conmedrl_continuous import RLTraining as ContinuousRLTraining


class _Policy:
    def rl_policy(self, state):
        return torch.zeros((len(state), 1), dtype=torch.long, device=state.device)


def test_fqe_single_state_confidence_interval_returns_three_values():
    fqe = FQE.__new__(FQE)
    fqe.eval_agent = _Policy()
    fqe.policy_net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        fqe.policy_net.weight.zero_()
        fqe.policy_net.bias.fill_(3.0)

    mean, upper, lower = fqe.avg_Q_value_est(torch.ones((1, 2)), 1.96)
    assert (mean, upper, lower) == (3.0, 3.0, 3.0)


def test_exponentiated_gradient_accepts_constraint_weights_plus_slack():
    for trainer_type in (RLTraining, ContinuousRLTraining):
        trainer = trainer_type.__new__(trainer_type)
        values = trainer.exponentiated_gradient(
            lambda_list=[1.0, 1.0, 1.0],
            constraint_violation_list=[0.2, -0.1],
            lr_list=[0.5, 0.5],
            B=3.0,
        )
        assert len(values) == 3
        assert math.isclose(sum(values), 3.0)
        assert all(value >= 0 for value in values)
        assert values[0] > values[2]
        assert values[1] < values[2]


def test_bounded_projected_gradient_enforces_orthant_and_l2_ball():
    for trainer_type in (RLTraining, ContinuousRLTraining):
        trainer = trainer_type.__new__(trainer_type)
        assert trainer.projected_gradient_update([-2.0, 3.0], B=5.0) == [0.0, 3.0]
        values = trainer.projected_gradient_update([-2.0, 3.0], B=2.0)
        assert values == [0.0, 2.0]
        assert all(value >= 0 for value in values)
        assert math.sqrt(sum(value * value for value in values)) <= 2.0
