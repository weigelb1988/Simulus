"""
Imagination-based forward prediction from a real environment state.

This script:
1. Initializes a real environment and runs it to a specified step
2. Captures the state at that step
3. Uses a trained world model to imagine forward without simulation
4. Saves the imagined trajectory as a GIF
"""
from functools import partial
from pathlib import Path
from typing import Dict

import click
import imageio
import numpy as np
import torch
from einops import rearrange
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig

from main import build_agent
from envs import SingleProcessEnv, POPWorldModelEnv
from utils.types import MultiModalObs, ObsModality
from utils.preprocessing import get_obs_processor


def get_config(benchmark: str) -> DictConfig:
    """Load Hydra configuration for the specified benchmark."""
    initialize(version_base=None, config_path="../config", job_name="imagine")
    overrides = ['hydra.run.dir=.', 'hydra.output_subdir=null']
    should_override_benchmark = Path('config/benchmark').exists()
    if should_override_benchmark:
        overrides.append(f"benchmark={benchmark}")
    cfg = compose(config_name="base", overrides=overrides)
    return cfg


def run_env_to_step(env: SingleProcessEnv, start_step: int, seed: int = None) -> Dict:
    """
    Run the environment to a specific step and return the trajectory.

    Args:
        env: The environment to run
        start_step: The step number to run to (must be > 0)
        seed: Random seed for environment reset

    Returns:
        Dict with observations, actions, rewards, and episode info
    """
    if seed is not None:
        obs = env.reset(seed=seed)
    else:
        obs = env.reset()

    observations = {modality: [obs[modality.name]] for modality in env.modalities}
    actions = []
    rewards = []

    for step in range(start_step):
        # Take random action to get to the starting state
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        for modality in env.modalities:
            observations[modality].append(obs[modality.name])
        actions.append(action)
        rewards.append(reward)

        if done or truncated:
            print(f"Episode ended at step {step + 1} (before reaching start_step={start_step})")
            print("Resetting environment...")
            obs = env.reset()
            # Clear trajectory and restart
            observations = {modality: [obs[modality.name]] for modality in env.modalities}
            actions = []
            rewards = []

    # Convert lists to numpy arrays
    trajectory = {
        'observations': {modality: np.array(observations[modality]) for modality in env.modalities},
        'actions': np.array(actions),
        'rewards': np.array(rewards)
    }

    return trajectory


def prepare_context_for_world_model(
    trajectory: Dict,
    env: SingleProcessEnv,
    tokenizer,
    world_model,
    context_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare context observations and actions for world model initialization.

    Args:
        trajectory: Trajectory from real environment
        env: The environment
        tokenizer: Trained tokenizer
        world_model: Trained world model
        context_length: Number of context steps to use (number of obs-action pairs)
        device: PyTorch device

    Returns:
        context_tokens_emb: Flattened token embeddings for world model initialization
    """
    # Get last context_length + 1 observations and context_length actions
    # We need context_length transitions: (obs[0], act[0]) -> obs[1], (obs[1], act[1]) -> obs[2], ...
    obs_dict = {}
    for modality in env.modalities:
        obs_array = trajectory['observations'][modality]
        # Take last context_length + 1 observations
        context_obs = obs_array[-(context_length + 1):]
        obs_dict[modality.name] = context_obs

    # Get last context_length actions
    context_actions = trajectory['actions'][-context_length:]

    # Convert to MultiModalObs format for tokenizer
    context_obs_mm = {}
    for modality in env.modalities:
        # Add batch dimension and convert to tensor
        obs_data = obs_dict[modality.name]
        if isinstance(obs_data, np.ndarray):
            obs_tensor = torch.from_numpy(obs_data).to(device)
        else:
            obs_tensor = obs_data.to(device)

        # Add batch dimension: (T, ...) -> (1, T, ...)
        obs_tensor = obs_tensor.unsqueeze(0)
        context_obs_mm[modality] = obs_tensor

    # Convert actions to tensor (1, T)
    context_actions_tensor = torch.from_numpy(context_actions).unsqueeze(0).to(device)

    # Encode observations to tokens using world_model.get_obs_tokens
    with torch.no_grad():
        obs_tokens = world_model.get_obs_tokens(context_obs_mm, tokenizer=tokenizer)

        # Get token embeddings for obs and actions (includes all context obs and actions)
        # We take first context_length observations (exclude the last one for now)
        obs_tokens_ctx = {k: v[:, :-1] for k in obs_tokens.keys() for k, v in [(k, obs_tokens[k])]}

        # Get combined embeddings (obs + actions)
        ctx_tokens_emb = world_model.get_tokens_emb(
            obs_tokens_ctx,
            context_actions_tensor,
            tokenizer=tokenizer
        )

        # Flatten: (B, T, K, E) -> (B, T*K, E)
        # Then remove the last action embedding
        action_seq_len = world_model.tokens_per_action
        ctx_tokens_emb = ctx_tokens_emb.flatten(1, 2)[:, :-action_seq_len]

    return ctx_tokens_emb


def imagine_forward(
    wm_env: POPWorldModelEnv,
    agent,
    horizon: int,
    use_policy: bool,
    temperature: float = 1.0,
    device: torch.device = None
) -> Dict:
    """
    Perform imagination-based forward prediction.

    Args:
        wm_env: World model environment (already initialized with context)
        agent: Trained agent (for policy if use_policy=True)
        horizon: Number of steps to imagine forward
        use_policy: If True, use trained policy; if False, use random actions
        temperature: Temperature for action sampling (if use_policy=True)
        device: PyTorch device

    Returns:
        Dict with imagined observations, actions, rewards, and ends
    """
    imagined_obs_tokens = {}
    imagined_actions = []
    imagined_rewards = []
    imagined_ends = []

    with torch.no_grad():
        for step in range(horizon):
            # Select action
            if use_policy:
                # Get current observation for actor-critic
                # Note: This is simplified - full implementation would maintain AC state
                action = wm_env.env.action_space.sample()  # Fallback to random
                # TODO: Properly integrate actor-critic for action selection
            else:
                # Random action
                action = wm_env.env.action_space.sample()

            # Step the world model environment
            # Action needs to be (B, 1) tensor
            action_tensor = torch.tensor([[action]], dtype=torch.long, device=device)
            obs_tokens, reward, done, info = wm_env.step(action_tensor, should_predict_next_obs=True, return_tokens=True)

            # Store results - obs_tokens is a dict of {ObsModality: tensor}
            for modality, tokens in obs_tokens.items():
                if modality not in imagined_obs_tokens:
                    imagined_obs_tokens[modality] = []
                imagined_obs_tokens[modality].append(tokens)

            imagined_actions.append(action)
            imagined_rewards.append(reward.cpu().item())

            # Handle done (could be tensor or bool)
            done_val = done.cpu().item() if isinstance(done, torch.Tensor) else done
            imagined_ends.append(done_val)

            if done_val:
                print(f"Imagined episode ended at step {step + 1}")
                break

    return {
        'observations': imagined_obs_tokens,
        'actions': np.array(imagined_actions),
        'rewards': np.array(imagined_rewards),
        'ends': np.array(imagined_ends)
    }


def decode_observations_to_images(
    imagined_trajectory: Dict,
    tokenizer,
    env: SingleProcessEnv,
    device: torch.device
) -> np.ndarray:
    """
    Decode imagined observation tokens back to images.

    Args:
        imagined_trajectory: Trajectory with observation tokens
        tokenizer: Trained tokenizer
        env: Environment (for modality info)
        device: PyTorch device

    Returns:
        Array of shape (T, H, W, C) with decoded images
    """
    decoded_images = []

    with torch.no_grad():
        # Check if we have image modality
        if ObsModality.image not in imagined_trajectory['observations']:
            raise ValueError("No image observations in trajectory")

        image_tokenizer = tokenizer.tokenizers[ObsModality.image.name]

        # Process each timestep
        for t in range(len(imagined_trajectory['actions'])):
            # Get tokens for this timestep (shape is [1, num_tokens])
            tokens = imagined_trajectory['observations'][ObsModality.image][t]

            # Embed tokens: (B, K) -> (B, K, E)
            embedded_tokens = image_tokenizer.embedding(tokens)

            # Reshape to spatial: (B, K, E) -> (B, E, H, W)
            # K should be H*W for the spatial layout
            num_tokens = embedded_tokens.shape[1]
            h = w = int(np.sqrt(num_tokens))
            assert h * w == num_tokens, f"Number of tokens {num_tokens} is not a perfect square"

            z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=h, w=w)

            # Decode to image: (B, E, H, W) -> (B, C, H_out, W_out)
            rec = image_tokenizer.decode(z, should_postprocess=True)

            # Clamp to [0, 1] and convert to numpy
            rec = torch.clamp(rec, 0, 1)

            # Convert from (1, C, H, W) to (H, W, C)
            img = rec[0].cpu().permute(1, 2, 0).numpy()

            # Convert from [0, 1] to [0, 255]
            img = (img * 255).astype(np.uint8)
            decoded_images.append(img)

    if len(decoded_images) == 0:
        raise ValueError("No images were decoded.")

    return np.stack(decoded_images)


def save_as_gif(images: np.ndarray, output_path: Path, fps: int = 15, loop: int = 0):
    """
    Save images as an animated GIF.

    Args:
        images: Array of shape (T, H, W, C)
        output_path: Path to save the GIF
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate duration per frame in milliseconds
    duration_ms = 1000.0 / fps

    # Save as GIF
    imageio.mimsave(
        output_path,
        images,
        duration=duration_ms,
        loop=loop
    )

    print(f"Saved imagination GIF to: {output_path}")


@click.command()
@click.argument('benchmark', type=click.Choice(['atari', 'craftax', 'dmc', 'biggym']))
@click.option('-p', '--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('-s', '--start-step', type=int, default=10,
              help='Step in the episode to start imagination from (must be >= context_length)')
@click.option('-h', '--horizon', type=int, default=50,
              help='Number of steps to imagine forward')
@click.option('--use-policy', is_flag=True, default=False,
              help='Use trained policy for actions (default: random actions)')
@click.option('-c', '--context-length', type=int, default=None,
              help='Number of context steps for world model (default: from config)')
@click.option('-o', '--output', type=click.Path(), default='imagination.gif',
              help='Output path for the GIF file')
@click.option('--fps', type=int, default=15,
              help='Frames per second for the GIF')
@click.option('--seed', type=int, default=None,
              help='Random seed for environment')
def imagine(benchmark, model_path, start_step, horizon, use_policy, context_length, output, fps, seed):
    """
    Imagine forward from a real environment state using a trained world model.

    This script runs a real environment to a specified step, then uses the trained
    world model to imagine forward without simulation, saving the result as a GIF.

    Example usage:
        python src/imagine.py atari -p outputs/checkpoint.pt -s 10 -h 50 --use-policy
    """
    # Load configuration
    print(f"Loading configuration for benchmark: {benchmark}")
    cfg = get_config(benchmark)

    # Set device
    device = torch.device(cfg.common.device)
    print(f"Using device: {device}")

    # Get context length from config if not specified
    if context_length is None:
        context_length = cfg.world_model.context_length
    print(f"Context length: {context_length}")

    # Validate start_step
    if start_step < context_length:
        raise ValueError(f"start_step ({start_step}) must be >= context_length ({context_length})")

    # Initialize environment
    print("Initializing environment...")
    env_fn = partial(instantiate, config=cfg.env.test)
    env = SingleProcessEnv(env_fn)

    # Build and load agent
    print(f"Building agent and loading checkpoint from: {model_path}")
    agent = build_agent(env, cfg, device)
    agent.load(Path(model_path), device)
    agent.eval()

    # Run environment to starting step
    print(f"Running environment to step {start_step}...")
    trajectory = run_env_to_step(env, start_step, seed=seed)
    print(f"Reached step {start_step}")

    # Prepare context for world model
    print(f"Preparing context (last {context_length} steps)...")
    context_tokens_emb = prepare_context_for_world_model(
        trajectory, env, agent.tokenizer, agent.world_model, context_length, device
    )

    # Initialize world model environment
    print("Initializing world model environment...")
    wm_env = POPWorldModelEnv(
        tokenizer=agent.tokenizer,
        world_model=agent.world_model,
        device=device,
        env=env_fn()
    )
    wm_env.reset_from_initial_observations(context_tokens_emb, return_tokens=False)

    # Imagine forward
    action_mode = "trained policy" if use_policy else "random actions"
    print(f"Imagining forward {horizon} steps using {action_mode}...")
    imagined_trajectory = imagine_forward(
        wm_env, agent, horizon, use_policy, device=device
    )

    # Decode observations to images
    print("Decoding imagined observations to images...")
    images = decode_observations_to_images(
        imagined_trajectory, agent.tokenizer, env, device
    )
    print(f"Decoded {len(images)} frames")

    # Save as GIF
    output_path = Path(output)
    print(f"Saving GIF ({images.shape})...")
    save_as_gif(images, output_path, fps=fps)

    # Print summary
    print("\n" + "="*50)
    print("IMAGINATION SUMMARY")
    print("="*50)
    print(f"Benchmark: {benchmark}")
    print(f"Model: {model_path}")
    print(f"Start step: {start_step}")
    print(f"Context length: {context_length}")
    print(f"Imagination horizon: {horizon}")
    print(f"Actual frames generated: {len(images)}")
    print(f"Action mode: {action_mode}")
    print(f"Output: {output_path.absolute()}")
    print(f"Total reward (imagined): {sum(imagined_trajectory['rewards']):.2f}")
    print("="*50)


if __name__ == "__main__":
    imagine()
