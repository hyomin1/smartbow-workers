import zmq, time


def get_pub_socket(port: int):
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)

    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.CONFLATE, 1)

    socket.bind(f"tcp://*:{port}")
    time.sleep(0.5)
    return socket


def get_sub_socket(port: int):
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)

    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
    socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 300)
    socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 300)
    socket.connect(f"tcp://127.0.0.1:{port}")

    return socket
