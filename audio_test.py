from pygame import mixer
# Initialize pygame mixer
mixer.init()
# Load the sounds
sound = mixer.Sound('applause-1.wav')
sound.play()
