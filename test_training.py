"""Quick test to verify training works"""
import sys
sys.path.insert(0, '/Users/robbymoseley/CascadeProjects/OaKPole/oak_cartpole')

from config import Config
from main import OaKAgent

# Override config for quick test
Config.NUM_EPISODES = 5
Config.EVAL_FREQ = 100  # Skip eval for speed

agent = OaKAgent(Config())
print("\n🚀 Starting OaK-CartPole test training (5 episodes)...\n")
agent.train(Config.NUM_EPISODES)
print("\n✅ Training completed successfully!")
print(f"Episode returns: {agent.episode_returns}")
