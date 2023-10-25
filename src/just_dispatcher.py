"""
pytorch 版分发端代码
"""
import os
import queue
import select
import socket
from threading import Thread
from typing import List

import torch

from node_state import socket_recv, socket_send

import lz4.frame
import zfpy

device = 'cpu'

class TestDispatcher:
    def __init__(self, nodes) -> None:
        self.nodes = nodes
        self.chunk_size = 512 * 1024
        # self.data_socket_port = 3019
        # early_exit_socket_port[-1] 替代了 data_socket_port
        self.early_exit_socket_port = [3021, 3022, 3023, 3024]
        pass

    # 用来压缩和解压 data
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))
    
    def _send_sub_models(self, sub_models: list):
        for i in range(len(sub_models)):
            model_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            model_client.setblocking(0)
            model_client.settimeout(100)

            model_client.connect(('localhost', self.nodes[i][0]))
            next_node_data_port = self.nodes[i+1][1] if i != len(sub_models) - 1 else self.early_exit_socket_port[-1]

            # 发模型，这里发了主干模型和早退模型
            # model_bytes = sub_models[i].save_to_buffer()
            main_model_bytes, exit_model_bytes = sub_models[i][0].save_to_buffer(), sub_models[i][1].save_to_buffer()
            socket_send(main_model_bytes, model_client, chunk_size=self.chunk_size)
            socket_send(exit_model_bytes, model_client, chunk_size=self.chunk_size)

            # 发端口，这里发了下个节点和早退回去的端口
            socket_send(str(next_node_data_port).encode(), model_client, chunk_size=1)
            socket_send(str(self.early_exit_socket_port[i]).encode(), model_client, chunk_size=1)
            select.select([model_client], [], []) # Waiting for acknowledgement: 0x06
            model_client.recv(1)

    def _data_client(self, input: queue.Queue):
        data_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # data_client.connect((self.computeNodes[0], 5000))
        # TODO: 本机实验
        data_client.connect(('localhost', self.nodes[0][1]))
        print("[DEBUG] data client connected, port ", self.nodes[0][1])
        data_client.setblocking(0)

        while True:
            model_input = input.get()
            out = self._comp(model_input)
            socket_send(out, data_client, self.chunk_size)
            print("[DEBUG] data client sent to nodes[0]")

    def _data_server(self, output: queue.Queue):
        inputs = []
        connected = []
        for port in self.early_exit_socket_port:
            data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_server.bind(('0.0.0.0', port))
            data_server.listen(1)
            data_server.setblocking(0)
            inputs.append(data_server)
        
        while True:
            readable, _, _ = select.select(inputs, [], [], 5)
            for sock in readable:
                if sock not in connected:
                    data_cli = sock.accept()[0]
                    print("[DEBUG] early exit result server accepted, port", port)
                    inputs.append(data_cli)
                    connected.append(data_cli)
                else:
                    data = bytes(socket_recv(sock, self.chunk_size))
                    print("result server received data")
                    pred = self._decomp(data)
                    print('pred:', pred)
                    output.put(pred)
    
    def run(self):
        sub_models = []
        SUB_MODEL_PATH = './sub_models/resnet50'
        for i in range(4):
            sub_models.append((
                torch.jit.load(os.path.join(SUB_MODEL_PATH, f'main_model_{i}.pth')).to(device),
                torch.jit.load(os.path.join(SUB_MODEL_PATH, f'exit_model_{i}.pth')).to(device),
            ))
        
        self._send_sub_models(sub_models)

        input_queue = queue.Queue(10)
        output_queue = queue.Queue(10)
        
        data_client_thread = Thread(target=self._data_client, args=(input_queue,))
        data_server_thread = Thread(target=self._data_server, args=(output_queue,)) 

        data_client_thread.start()
        data_server_thread.start()

        # 修改一下 input_shape
        x = torch.randn(1, 3, 32, 32).numpy()
        for _ in range(1000):
            input_queue.put(x)

        data_client_thread.join()
        data_server_thread.join()


dispatcher = TestDispatcher([
    (3001, 3011), (3002, 3012), (3003, 3013), (3004, 3014)
])
dispatcher.run()