import datetime
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' if you have PyQt/PySide installed
from model import Model

model = Model({ 
    'steps': 200,

    'train': True,
    'render': False,
    'policy_dir': 'checkpoints',
    'autosave_interval': 60,

    'record_gif': True,
    'gif_path': f'./simulation_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.gif',
    'gif_fps': 20,
})
model.run()