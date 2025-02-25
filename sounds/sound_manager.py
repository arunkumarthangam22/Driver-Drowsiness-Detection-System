import pygame
import time
from threading import Thread

class SoundManager:
    def __init__(self):
        pygame.mixer.init()

    def play_sound(self, sound_path, loop=False):
        """Plays a sound. If loop is True, it plays indefinitely."""
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1 if loop else 0)

    def stop_sound(self):
        """Stops the currently playing sound."""
        pygame.mixer.music.stop()
        time.sleep(2)

    def play_message(self, sound_path, message):
        """Plays an alert message once."""
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            print(message)  # Placeholder for UI text display
            continue
