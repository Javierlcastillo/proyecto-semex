from traffic_light import TrafficLight, TLConnection
from defined_routes import routes

tlconnections: list[TLConnection] = []

tl1 = TrafficLight(270, 250)
tlc1 = TLConnection(routes[0], 100, tl1)
tlc2 = TLConnection(routes[1], 500, tl1)

tlconnections.append(tlc1)