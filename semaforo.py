import agentpy as ap

class Semaforo(ap.Agent):
    def setup(self, **kwargs):
        self.estado = kwargs.get("estado", "rojo")
        self.tiempo_rojo = kwargs.get("tiempo_rojo", 30)
        self.tiempo_verde = kwargs.get("tiempo_verde", 30)
        self.tiempo_amarillo = kwargs.get("tiempo_amarillo", 5)
        self.temporizador = self.tiempo_rojo
        super().setup(**kwargs)

    def cambiar_estado(self):
        if self.estado == "rojo":
            self.estado = "verde"
            self.temporizador = self.tiempo_verde
        elif self.estado == "verde":
            self.estado = "amarillo"
            self.temporizador = self.tiempo_amarillo
        elif self.estado == "amarillo":
            self.estado = "rojo"
            self.temporizador = self.tiempo_rojo

    def actualizar(self):
        self.temporizador -= 1
        if self.temporizador <= 0:
            self.cambiar_estado()

    def reiniciar(self):
        self.estado = "rojo"
        self.temporizador = self.tiempo_rojo

    def obtener_estado(self):
        return self.estado

    def __str__(self):
        return f"Semáforo(estado={self.estado}, temporizador={self.temporizador})"
    
