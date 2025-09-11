import datetime
import os

from model import Model

# Create videos directory if it doesn't exist
os.makedirs('./videos', exist_ok=True)

model = Model({ 
    'steps': 500,
    'train': True, # Por alguna razón, el código funciona mejor si está entrenando. Incluso cua

    'render_every': 10, # We will render only for debugging purposes, so not now.

    'policy_dir': 'checkpoints',
    'autosave_interval': 120,

    "offline_export": True,

    'record_gif': True,
    'gif_path': f'./videos/simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 27,
})

model.run() # type: ignore