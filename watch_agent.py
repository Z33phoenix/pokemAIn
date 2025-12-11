"""
Legacy viewer for quickly rendering the agent's 96x96 observations.
Includes a debug overlay for Warps (Red) and NPCs (Green).
"""

import cv2
import numpy as np
import yaml
import os
import sys

from src.core.env_factory import create_environment

def load_config():
    """Load the project config to check for debug flags."""
    config_path = os.path.join("config", "hyperparameters.yaml")
    if not os.path.exists(config_path):
        print("Config not found, using defaults.")
        return {"debug": False, "environment": {}}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def draw_overlay(img, info):
    """
    Draws debug boxes on the 256x256 viewer image.
    Warps -> Red (BGR: 0, 0, 255)
    NPCs  -> Green (BGR: 0, 255, 0)
    """
    # Player's global coordinates
    px, py = info.get("x", 0), info.get("y", 0)
    
    # Viewer Dimensions
    view_w, view_h = 256, 256
    
    # Game Screen Dimensions (Tiles)
    # The GameBoy screen is 10 tiles wide (160px) and 9 tiles high (144px).
    # We map this 10x9 grid onto the 256x256 viewer.
    tiles_visible_x = 10
    tiles_visible_y = 9
    
    # Pixels per tile in the 256x256 viewer
    tile_w = view_w / tiles_visible_x  # ~25.6 pixels
    tile_h = view_h / tiles_visible_y  # ~28.4 pixels
    
    # Center of the viewer (approximate player position)
    center_x = view_w / 2
    center_y = view_h / 2

    def draw_box(grid_x, grid_y, color):
        # Calculate relative distance from player
        dx = grid_x - px
        dy = grid_y - py
        
        # Calculate screen position (Center + Relative Offset)
        # We subtract half a tile size from top-left calculation so the box is centered on the grid point
        screen_x = int(center_x + (dx * tile_w) - (tile_w / 2))
        screen_y = int(center_y + (dy * tile_h) - (tile_h / 2))
        
        top_left = (screen_x, screen_y)
        bottom_right = (int(screen_x + tile_w), int(screen_y + tile_h))
        
        # Draw only if arguably on screen to avoid weird wrapping
        if -50 < screen_x < 300 and -50 < screen_y < 300:
            cv2.rectangle(img, top_left, bottom_right, color, 2)

    # 1. Draw Warps (Red)
    for warp in info.get("map_warps", []):
        draw_box(warp["x"], warp["y"], (0, 0, 255))

    # 2. Draw Sprites (Green)
    for sprite in info.get("sprites", []):
        draw_box(sprite["x"], sprite["y"], (0, 255, 0))

    return img

def main() -> None:
    """Run a random-policy viewer that displays the agent's observation."""
    cfg = load_config()
    env_cfg = cfg.get("environment", {}).copy()

    # Force windowed mode for the viewer
    env_cfg["headless"] = False
    env_cfg["rom_path"] = env_cfg.get("rom_path", "pokemon_red.gb")

    # Use environment factory (supports hot-swapping between games)
    game_id = cfg.get("game", "pokemon_red")
    env = create_environment(game_id, env_cfg)
    debug_mode = cfg.get("debug", False)

    obs, info = env.reset()
    print(f"Viewer started. Debug Overlay: {'ON' if debug_mode else 'OFF'}")
    print("Press 'q' in the OpenCV window to quit.")

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # 1. Get the raw 96x96 observation
        brain_view = obs[0]
        
        # 2. Resize to 256x256 for human visibility
        brain_view_big = cv2.resize(brain_view, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # 3. Convert to BGR so we can draw colored boxes (OpenCV expects BGR)
        brain_view_color = cv2.cvtColor(brain_view_big, cv2.COLOR_GRAY2BGR)

        # 4. Draw Overlay if Debug is enabled
        if debug_mode:
            brain_view_color = draw_overlay(brain_view_color, info)

        cv2.imshow("What the AI Sees", brain_view_color)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()