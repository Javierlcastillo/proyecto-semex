import socket
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM);
s.bind(('127.0.0.1', 1101))
s.listen()

while True:
    conn, addr = s.accept()
    conn.send(b"I will send key $")
    data = conn.recv(4096)
    if not data: break
    print ("Received from client: ",data.decode("ascii"))
    
'''
connection,address=s.accept();
with connection:
    a=connection.recv(1024);
    print(a.decode("ascii"))
    connection.close();
'''