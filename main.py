import datetime
import matplotlib
import os
matplotlib.use('Agg')  # or 'QtAgg' if you have PyQt/PySide installed
from model import Model

# Create videos directory if it doesn't exist
os.makedirs('./videos', exist_ok=True)

model = Model({ 
    'steps': 2000,
    'train': False,
    'render': False,
    'policy_dir': 'checkpoints',
    'autosave_interval': 60,

    'record_gif': True,
    'gif_path': f'./videos/simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 20,

    'flow_path': 'traffic_flow.json',
})
model.run()