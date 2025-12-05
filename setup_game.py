import sys
import os
import pyboy
import traceback

def main():
    # 1. Verify ROM Exists
    rom_path = 'pokemon_red.gb'
    if not os.path.exists(rom_path):
        print(f"\n[ERROR] Could not find '{rom_path}'")
        print(f"Current Working Directory: {os.getcwd()}")
        print("Please ensure the ROM file is in this directory.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"[INFO] Found ROM: {rom_path}")
    print("[INFO] Initializing PyBoy...")

    try:
        # 2. Initialize PyBoy
        # Note: We removed window_scale as it causes crashes in your version
        pb = pyboy.PyBoy(rom_path, window="SDL2")
        pb.set_emulation_speed(1)
        
        print("[INFO] Game Window Opened.")
        print("Instructions:")
        print("   - Play until you are in the bedroom.")
        print("   - CLOSE THE WINDOW (Click X) to save the state.")

        # 3. Game Loop
        # tick() returns True while the game is running, False when window is closed
        while pb.tick():
            pass

        # 4. Save State on Exit
        print("\n[INFO] Window closed. Saving 'initial.state'...")
        with open("initial.state", "wb") as f:
            pb.save_state(f)
        print("[SUCCESS] State saved! You can now run the agent.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] The game crashed: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()