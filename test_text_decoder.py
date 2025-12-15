#!/usr/bin/env python3
"""
Quick test script for text_decoder fix validation.

This script loads a Pokemon GB game and runs through some dialogue
to validate that the text_decoder is correctly capturing complete
dialogue text (both single-page and multi-page).
"""

import sys
import io

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable video for headless testing

from pyboy import PyBoy
from src.games.gb.text_decoder import TextDecoder

def test_text_decoder():
    """Load game and test text decoder with actual dialogue."""

    # Initialize PyBoy with Pokemon Red
    game_path = 'src/games/gb/pokemon_red.gb'
    if not os.path.exists(game_path):
        print(f"ERROR: Game file not found: {game_path}")
        return

    print(f"Loading game from: {game_path}")
    pyboy = PyBoy(game_path, window_type='headless')

    # Initialize text decoder
    text_decoder = TextDecoder(pyboy)

    print("\nRunning game simulation...")
    print("This will advance frames and check for dialogue...")
    print("Collecting both complete and partial dialogues to verify text reading...")

    try:
        last_complete_narrative = ''
        last_raw_narrative = ''
        complete_dialogue_count = 0
        partial_sightings = 0
        max_frames = 8000  # Run for longer to see more dialogue

        for frame_num in range(max_frames):
            # Advance game
            pyboy.tick()

            # Decode text - disable verbose debug for cleaner output
            result = text_decoder.decode(debug=False)
            complete_narrative = result.get('narrative', '').strip()

            # Also manually read raw narrative from tile map to see partial text
            tile_map = text_decoder._get_tile_map(use_window_layer=True)
            raw_narrative = text_decoder.read_narrative(tile_map, debug=False).strip()

            # Track completed narratives (from our dialogue detection logic)
            if complete_narrative and complete_narrative != last_complete_narrative:
                complete_dialogue_count += 1
                # Check if this dialogue is separated correctly (no mixing different sources)
                is_clean = not ('MOM:' in complete_narrative and 'SNES' in complete_narrative)
                status = "✓ CLEAN" if is_clean else "✗ MIXED"
                print(f"\n[Frame {frame_num}] COMPLETE Dialogue #{complete_dialogue_count} {status}:")
                print(f"  Text: '{complete_narrative}'")
                print(f"  Length: {len(complete_narrative)}")
                last_complete_narrative = complete_narrative

            # Track raw narrative changes (to see if text IS being read)
            if raw_narrative and raw_narrative != last_raw_narrative and len(raw_narrative) >= 5:
                partial_sightings += 1
                if partial_sightings <= 10:  # Only print first 10 partial sightings to avoid spam
                    print(f"[Frame {frame_num}] Raw text sighting: '{raw_narrative}'")
                last_raw_narrative = raw_narrative

            # Press button periodically to advance dialogue
            if frame_num % 30 == 0 and frame_num > 100:
                pyboy.button_press('a')
            elif frame_num % 30 == 15 and frame_num > 100:
                pyboy.button_release('a')

            # Stop early if we've seen enough complete dialogues
            if complete_dialogue_count >= 6:
                print(f"\nCollected {complete_dialogue_count} complete dialogues, stopping test.")
                break

        print(f"\n✓ Test completed!")
        print(f"✓ Complete dialogues collected: {complete_dialogue_count}")
        print(f"✓ Partial text sightings: {partial_sightings}")
        if partial_sightings > 0 and complete_dialogue_count == 0:
            print("⚠ Note: Text IS being read, but dialogue completion detection may need adjustment")

    finally:
        pyboy.stop()

if __name__ == '__main__':
    test_text_decoder()
