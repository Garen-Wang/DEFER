"""
pytorch 版用来跑早退点切分模型的节点端模型
"""
import argparse
import io
from queue import Queue
import queue
import select
import socket
from threading import Thread
import time

import torch
import torch.nn as nn
from node_state import NodeState, socket_recv, socket_send

import zfpy
import lz4.frame

device = 'cpu'

class TestNode:
    def __init__(self, model_socket_port: int, data_socket_port: int) -> None:
        self.model_socket_port = model_socket_port
        self.data_socket_port = data_socket_port

    def _model_socket(self, node_state: NodeState):
        model_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        model_server.bind(('0.0.0.0', self.model_socket_port))
        print("[DEBUG] Model socket running, port ", self.model_socket_port)
        model_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        model_server.listen(1) 
        model_cli = model_server.accept()[0]
        print("[DEBUG] Model socket accepted")
        model_cli.setblocking(0)
        model_cli.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)

        # 收模型，主干模型和早退模型
        main_model_bytes = socket_recv(model_cli, node_state.chunk_size)
        exit_model_bytes = socket_recv(model_cli, node_state.chunk_size)

        # 收端口号，下个节点和早退端点
        # TODO: 本机实验，next_node 暂时先传成了端口号
        next_node_data_port = socket_recv(model_cli, chunk_size=1)
        print("[DEBUG] Model socket received architecture & weights")
        dispatcher_early_exit_port = socket_recv(model_cli, chunk_size=1)

        # 加载模型
        main_model = torch.jit.load(io.BytesIO(main_model_bytes))
        main_model.eval()
        exit_model = torch.jit.load(io.BytesIO(exit_model_bytes))
        exit_model.eval()
        node_state.model = main_model
        node_state.model2 = exit_model

        node_state.next_node = int(next_node_data_port.decode())
        print("[DEBUG] model socket: next_node is ", node_state.next_node)
        node_state.dispatcher_port = int(dispatcher_early_exit_port.decode())
        print("[DEBUG] model socket: dispatcher EARLY EXIT port is ", node_state.dispatcher_port)
        select.select([], [model_cli], [])
        model_cli.send(b'\x06')
        model_server.close()
        print("[DEBUG] Model socket closed")
        print("[DEBUG] _model_socket Thread Finished")
    
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))

    # 发给下一个
    def _data_server_socket(self, node_state: NodeState, to_send: Queue):
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_server.bind(('0.0.0.0', self.data_socket_port))
        data_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        print("[DEBUG] Data server socket running")
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
        print("[DEBUG] Data server socket accepted")
        data_cli.setblocking(0)

        while True:
            data = bytes(socket_recv(data_cli, node_state.chunk_size))
            temp = self._decomp(data)
            to_send.put(temp)
        
        print("[DEBUG] Data server socket closed")
        print("[DEBUG] Data server socket Thread Finished")

    # 接收前一个发来的
    def _data_client_socket(self, node_state: NodeState, to_send: Queue):
        while node_state.next_node == '' or node_state.dispatcher_port == '':
            time.sleep(5)
        
        print("[DEBUG] Data client socket received next_node: ", node_state.next_node)
        main_model = node_state.model
        exit_model = node_state.model2

        next_node_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # next_node_client.connect((node_state.next_node, 5000))
        # TODO: 本机实验，测试完记得改回来
        next_node_client.connect(('localhost', node_state.next_node))
        print("[DEBUG] Data client socket connected, port", node_state.next_node)
        next_node_client.setblocking(0)

        is_final = (node_state.next_node == node_state.dispatcher_port)
        if not is_final:
            dispatcher_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dispatcher_client.connect(('localhost', node_state.dispatcher_port))
            print("[DEBUG] Early exit data client socket connected, port", node_state.dispatcher_port)
            dispatcher_client.setblocking(0)

        def is_early_exit(x) -> bool:
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk)
            return top1 > 0.5

        while True:
            # 在这里做节点端的推理
            with torch.no_grad():
                inputs = torch.from_numpy(to_send.get()).to(device)
                outputs = main_model(inputs)
                results = exit_model(outputs)
                # 判断是否能早退
                if not is_final and is_early_exit(results):
                    print("[DEBUG] EARLY EXIT!")
                    results_data = self._comp(results.detach().numpy())
                    socket_send(results_data, dispatcher_client, node_state.chunk_size)
                    print("[DEBUG] EARLY EXIT result sent back to dispatcher!!!")
                    continue

                print("[DEBUG] data client socket inference finished")
                output_data = self._comp(outputs.detach().numpy() if not is_final else results.detach().numpy())
                socket_send(output_data, next_node_client, node_state.chunk_size)
                print("[DEBUG] data client socket result sent to next node")

        print("[DEBUG] Data client socket closed")
        print("[DEBUG] Data client socket Thread Finished")

    def run(self):
        node_state = NodeState(chunk_size=512*1024)
        to_send = queue.Queue(1000)
        model_thread = Thread(target=self._model_socket, args=(node_state,))
        data_server_thread = Thread(target=self._data_server_socket, args=(node_state, to_send))
        data_client_thread = Thread(target=self._data_client_socket, args=(node_state, to_send))

        model_thread.start()
        data_server_thread.start()
        data_client_thread.start()
        model_thread.join()
        data_server_thread.join()
        data_client_thread.join()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script with argparse')
        # 添加model_port参数，默认值为3001
    parser.add_argument('--model_port', type=int, default=3001, help='Port for the model (default: 3001)')
    
    # 添加data_port参数，默认值为3002
    parser.add_argument('--data_port', type=int, default=3011, help='Port for the data (default: 3002)')
    args = parser.parse_args()

    model_port = args.model_port
    data_port = args.data_port

    node = TestNode(model_port, data_port)
    node.run()
