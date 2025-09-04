from traffic_light import TrafficLight, TLConnection
from defined_routes import routes

tlconnections: list[TLConnection] = []

tl1 = TrafficLight(270, 250)
tlc1 = TLConnection(routes[0], tl1, 100)

tlconnections.append(tlc1)