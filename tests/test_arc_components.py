import numpy as np
import torch

from encoders.state_encoder import ArcVisualContextEncoder, EncoderConfig
from environments.arc.env import ARCEnvironment
from models.hierarchical_policy import HierarchicalActionHead
from models.reward_model import ArcRewardModel
from environments.arc.options import FillRegionOption


def _synthetic_state(env: ARCEnvironment, target_color=3):
    num_colors = env.NUM_COLORS
    grid_size = env.working_grid_size
    grid_flat = np.full(grid_size * grid_size, target_color / (num_colors - 1), dtype=np.float32)
    spatial_feats = np.zeros(env._spatial_feature_dim, dtype=np.float32)
    exemplar_dim = 2 * num_colors + 3
    task = np.zeros(env.max_training_examples * exemplar_dim, dtype=np.float32)
    hist = np.zeros(num_colors, dtype=np.float32)
    hist[target_color] = 1.0
    task[num_colors: 2 * num_colors] = hist
    context = np.zeros(8, dtype=np.float32)
    return np.concatenate([grid_flat, spatial_feats, task, context]).astype(np.float32)


def _build_encoder(env: ARCEnvironment):
    num_colors = env.NUM_COLORS
    grid_size = env.working_grid_size
    max_examples = env.max_training_examples
    state_dim = env.state_dim
    config = EncoderConfig(
        input_dim=state_dim,
        latent_dim=128,
        working_grid_size=grid_size,
        num_colors=num_colors,
        max_training_examples=max_examples,
        exemplar_feature_dim=2 * num_colors + 3,
        num_attention_heads=4,
        transformer_layers=2,
    )
    return ArcVisualContextEncoder(config)


def test_arc_encoder_returns_tokens():
    env = ARCEnvironment()
    state = _synthetic_state(env)
    encoder = _build_encoder(env)

    latent, tokens = encoder.encode_with_tokens(torch.from_numpy(state))

    assert latent.shape[-1] == encoder.latent_dim
    assert 'grid' in tokens and tokens['grid'].shape[-1] == encoder.latent_dim
    assert tokens['grid'].shape[-2] == env.working_grid_size * env.working_grid_size


def test_hierarchical_action_head_generates_candidates():
    env = ARCEnvironment()
    env.grid_height = env.working_grid_size
    env.grid_width = env.working_grid_size
    env._rebuild_action_table()

    state = _synthetic_state(env)

    encoder = _build_encoder(env)
    head = HierarchicalActionHead(encoder, env)

    candidates = head.generate_candidates(state)
    assert len(candidates) > 0

    mapped = [env.action_index_from_structured(vec) for vec in candidates]
    assert any(idx is None or 0 <= idx < env.action_dim for idx in mapped)


def test_arc_reward_model_shaping_and_update():
    env = ARCEnvironment()
    state = _synthetic_state(env)
    encoder = _build_encoder(env)

    latent = torch.from_numpy(encoder.encode_numpy(state)).float()
    reward_model = ArcRewardModel(encoder.latent_dim)

    shaping = reward_model.shaping(latent, latent)
    assert abs(shaping) < 1e-5

    target = torch.tensor([0.7], dtype=torch.float32)
    loss = reward_model.update(latent, target)
    assert loss >= 0.0


def test_fill_region_option_aligns_to_exemplar():
    env = ARCEnvironment()
    state = _synthetic_state(env, target_color=5)
    option = FillRegionOption(
        -1,
        'fill_exemplar',
        env.state_dim,
        action_dim=1,
        exemplar_idx=0,
        max_training_examples=env.max_training_examples,
        spatial_feature_dim=env._spatial_feature_dim,
    )

    achieved, _ = option.termination(state)
    assert achieved
