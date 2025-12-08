"""
Legacy viewer for quickly rendering the agent's 96x96 observations.

This is not part of the training flow and exists only as a simple manual
visualization stub. The environment now requires a config dictionary; if you
need a viewer, prefer wiring a real agent into the loop or use TensorBoard
images. This file is kept for reference and may not be wired to your current
checkpoints.
"""

import cv2
import numpy as np

from src.env.pokemon_red_gym import PokemonRedGym


def main() -> None:
    """Run a random-policy viewer that displays the agent's 96x96 observation."""
    # Minimal config so the env boots in windowed mode. Adjust paths as needed.
    env = PokemonRedGym({"headless": False, "emulation_speed": 1, "rom_path": "pokemon_red.gb"})

    obs, info = env.reset()
    print("Random agent is playing... Press 'q' in the OpenCV window to quit.")

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        brain_view = obs[0]
        brain_view_big = cv2.resize(brain_view, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("What the AI Sees", brain_view_big)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
