import agentpy as ap

class Semaforo(ap.Agent):
    def setup(self, **kwargs):
        """
        Inicializa el agente semáforo con parámetros opcionales usando kwargs.
        """
        self.estado = kwargs.get("estado", "rojo")
        self.tiempo_rojo = kwargs.get("tiempo_rojo", 30)
        self.tiempo_verde = kwargs.get("tiempo_verde", 30)
        self.tiempo_amarillo = kwargs.get("tiempo_amarillo", 5)
        self.temporizador = self.tiempo_rojo
        # Llama a setup del padre si es necesario
        super().setup(**kwargs)
    
