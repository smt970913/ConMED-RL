import unittest

import numpy as np
import pandas as pd
import torch

from ConMedRL.conmedrl_continuous import RLConfig_custom, RLTraining
from ConMedRL.data_loader import TrainDataLoader, ValTestDataLoader


class ContinuousIntegrationTest(unittest.TestCase):
    def test_vector_actions_complete_actor_critic_fqe_step(self):
        rng = np.random.default_rng(7)
        rows, state_dim, action_dim = 32, 5, 2
        states = pd.DataFrame(rng.random((rows, state_dim)))
        outcome = pd.DataFrame(
            {
                "dose_a": rng.random(rows),
                "dose_b": rng.random(rows) * 2.0,
                "obj_cost": rng.random(rows),
                "con_cost_0": rng.random(rows),
                "stay_id": np.repeat(np.arange(4), 8),
                "subject_id": np.repeat(np.arange(4), 8),
                "row_index": np.arange(rows),
            }
        )
        outcome["done"] = 0.0
        outcome.loc[7::8, "done"] = 1.0

        config = RLConfig_custom(
            algo_name="continuous-test",
            gamma=0.99,
            batch_size=8,
            train_eps=1,
            train_eps_steps=1,
            lr_actor=1e-4,
            lr_critic=1e-4,
            weight_decay_actor=0.0,
            weight_decay_critic=0.0,
            optim_actor="torch.optim.Adam",
            optim_critic="torch.optim.Adam",
            loss_critic="nn.MSELoss()",
            activation_function_actor="relu",
            activation_params_actor={},
            activation_function_critic="relu",
            activation_params_critic={},
            weight_decay_fqe=0.0,
            optim_fqe="torch.optim.Adam",
            loss_fqe="nn.MSELoss()",
            activation_function_fqe="relu",
            activation_params_fqe={},
            memory_capacity=64,
            target_update=1,
            tau=0.01,
            lr_fqe_obj=1e-4,
            lr_fqe_con_list=[1e-4],
            lr_lambda_list=[1e-4],
            constraint_num=1,
            threshold_list=[0.2],
            device_type="cpu",
        )
        terminal = np.zeros(state_dim, dtype=np.float32)
        train_loader = TrainDataLoader(config, outcome, states, terminal)
        train_loader.data_buffer_train(
            ["dose_a", "dose_b"], num_constraint=1
        )
        val_loader = ValTestDataLoader(
            config,
            outcome.iloc[:8].copy(),
            states.iloc[:8].copy(),
            outcome,
            states,
            terminal,
        )
        val_loader.data_buffer(["dose_a", "dose_b"], num_constraint=1)

        training = RLTraining(
            config,
            state_dim,
            action_dim,
            train_loader.data_torch_loader_train,
            val_loader.data_torch_loader,
            action_bounds=[(0.0, 1.0), (0.0, 2.0)],
        )
        critic = training.critic_agent_config([8])
        actor = training.actor_agent_config(critic, [8])
        fqe = training.fqe_agent_config(actor, [8], eval_target="obj")
        constraint_fqe = training.fqe_agent_config(actor, [8], eval_target=0)

        batch = train_loader.data_torch_loader_train()
        state, action, objective, constraints, next_state, done = batch
        state_action = torch.cat((state, action), dim=1)
        critic_loss = critic.update(
            actor, [0.0], state_action, objective, constraints, next_state, done
        )
        actor_loss = actor.update(state)
        fqe_loss = fqe.update(state_action, objective, next_state, done)
        constraint_loss = constraint_fqe.update(
            state_action, constraints[0], next_state, done
        )

        self.assertEqual(action.shape, (8, 2))
        self.assertTrue(
            all(
                np.isfinite(value)
                for value in (critic_loss, actor_loss, fqe_loss, constraint_loss)
            )
        )
        predicted = actor.rl_policy(state)
        self.assertTrue(torch.all((predicted[:, 0] >= 0) & (predicted[:, 0] <= 1)))
        self.assertTrue(torch.all((predicted[:, 1] >= 0) & (predicted[:, 1] <= 2)))


if __name__ == "__main__":
    unittest.main()
