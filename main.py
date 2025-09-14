import datetime
import os

from model import Model

# Create videos directory if it doesn't exist
os.makedirs('./videos', exist_ok=True)


model = Model({ 
    'steps': 300,
    'train': True, # Por alguna razón, el código funciona mejor si está entrenando. Incluso cua
    'tl_mode': "qlearning",      # << choose fixed/qlearning

    'render_every': 10, # We will render only for debugging purposes, so not now.
    'plot_pad': 120,    # more/less margin
    'zoom_out': 1.4,    # >1 zooms out, <1 zooms in
                        # or fix exact box:
                        # plot_bounds = [0, 900, -100, 800]


    'policy_dir': 'checkpoints',
    'autosave_interval': 30,

    "offline_export": True,

    'record_gif': True,
    'gif_path': f'./videos/simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 27,
})

model.run() # type: ignore
