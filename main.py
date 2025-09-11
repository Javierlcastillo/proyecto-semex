import datetime
import os

from model import Model

# Create videos directory if it doesn't exist
os.makedirs('./videos', exist_ok=True)

model = Model({ 
    'steps': 200,
    'train': False,

    'render_every': 10,

    'policy_dir': 'checkpoints',
    'autosave_interval': 60,

    'record_gif': True,
    'gif_path': f'./videos/simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 20,
})

model.run() # type: ignore