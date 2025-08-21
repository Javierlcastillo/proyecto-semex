import agentpy as ap
import pandas as pd

class Carro(ap.Agent):
    def setup(self, **kwargs):
        pos_inicial = { 'mov1': (x, y, z),
                        'mov2': (x, y, z), 
                        'mov3': (x, y, z), 
                        'mov4': (x, y, z),
                        'mov5': (x, y, z),
                        'mov6': (x, y, z),
                        'mov7': (x, y, z),
                        'mov8': (x, y, z),
                        'mov9': (x, y, z),
                        'mov10': (x, y, z),
                        'mov11': (x, y, z),
                        'mov12': (x, y, z),
                        'mov13': (x, y, z),
                        'mov14': (x, y, z)
                    }

        return super().setup(**kwargs)
    

    
kwargs  = { 'movimiento': None, 'tipo': None, }