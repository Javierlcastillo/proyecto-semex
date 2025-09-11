import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' if you have PyQt/PySide installed
from model import Model

model = Model({ 
    'steps': 2000,

    'train': False,
    'render': True,
    'policy_dir': 'checkpoints',
    'autosave_interval': 60,
})
model.run()