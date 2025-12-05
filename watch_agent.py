import time
import cv2
import numpy as np
from src.env.pokemon_red_gym import PokemonRedGym

def main():
    # 1. Init Environment with headless=False to see the PyBoy window
    # emulation_speed=1 ensures it plays at real-time (60fps) rather than 1000fps
    env = PokemonRedGym(headless=False, emulation_speed=1)
    
    obs, info = env.reset()
    
    print("Agent is playing... Press 'q' in the OpenCV window to quit.")

    while True:
        # 2. Get Action (Random for now, replace with: agent.predict(obs))
        # action = agent.predict(obs) 
        action = env.action_space.sample() 
        
        # 3. Step the Environment
        obs, reward, done, truncated, info = env.step(action)
        
        # 4. Visualize the "Brain's Eye"
        # obs is shape (1, 84, 84). We remove the channel dim to get (84, 84)
        brain_view = obs[0] 
        
        # Resize it up to 256x256 so it's big enough to see on your monitor
        brain_view_big = cv2.resize(brain_view, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Display the "Brain View" in a separate OpenCV window
        cv2.imshow("What the AI Sees", brain_view_big)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()