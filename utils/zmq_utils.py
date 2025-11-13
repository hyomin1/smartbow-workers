import zmq


def get_pub_socket(port: int):
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    return socket


def get_sub_socket(ports: list[int]):
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    for p in ports:
        socket.connect(f"tcp://127.0.0.1:{p}")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    return socket
