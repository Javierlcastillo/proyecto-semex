import socket
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1101))
from_server = s.recv(4096)
print ("Received from server: ",from_server.decode("ascii"))
s.send(b"I have received ZZZZ msg")

'''
data=s.send(b'Hola XYZ Message$')
s.close()
'''