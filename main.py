import datetime
import os

from model import Model

# Create videos directory if it doesn't exist
os.makedirs('./videos', exist_ok=True)

model = Model({ 
    'steps': 3000,
    'train': True,

    'render_every': 0,

    'policy_dir': 'checkpoints',
    'autosave_interval': 120,

    'record_gif': True,
    'gif_path': f'./videos/simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 27,
})

model.run() # type: ignore