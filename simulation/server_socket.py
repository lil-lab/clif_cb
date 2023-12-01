"""ServerSocket, used for creating/destroying and communicating with the Unity standalone."""
from __future__ import annotations

import logging
import platform
import signal
import socket
import sys

from threading import Thread
from time import sleep
from subprocess import Popen, call

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from protobuf import CerealBarProto_pb2

WINDOWS_OS: str = 'Windows'
LINUX_OS: str = 'Linux'
MAC_OS: str = 'Darwin'

WINDOWS_EXEC_NAME = 'Cereal-Bar'
LINUX_EXEC_NAME = './linux-build/Cereal-Bar.x86_64'
MAC_EXEC_NAME = 'Cereal-Bar.app'


def start_standalone() -> Popen:
    """ Starts the standalone process and returns a Popen object representing the new process.

    Raises:
        ValueError, if the OS is unrecognized.
    """
    system_os: str = platform.system()
    logging.info('Starting standalone with system OS: ' + system_os)

    if system_os == WINDOWS_OS:
        return Popen(WINDOWS_EXEC_NAME)
    elif system_os == LINUX_OS:
        return Popen(LINUX_EXEC_NAME)
    elif system_os == MAC_OS:
        return Popen(["open", MAC_EXEC_NAME])
    else:
        raise ValueError('Unrecognized OS: ' + system_os)


def quit_standalone(old_process: Popen) -> None:
    """ Quits an existing Popen process.

    Raises:
        ValueError, if the OS is unrecognized.
    """
    old_process.kill()

    system_os: str = platform.system()
    if system_os == WINDOWS_OS or system_os == LINUX_OS:
        old_process.wait()
    elif system_os == MAC_OS:
        call(['osascript', '-e', 'quit app "' + MAC_EXEC_NAME + '"'])
    else:
        raise ValueError('Unrecognized OS: ' + system_os)


class ServerSocket:
    """ Stores the sockets connecting to the Unity game."""
    def __init__(self, ip_address: str, port: int):
        self._ip_address: str = ip_address
        self._port: int = port

        self._server_socket: socket.SocketKind = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                                       1)
        self._server_socket.bind((self._ip_address, self._port))
        self._server_socket.listen(5)

        print('Listening at ' + str(self._ip_address) + ':' + str(self._port))
        self.client_socket: socket.SocketKind = None

        self._current_process: Popen = None

    def _accept_connection(self) -> None:
        try:
            client: Tuple[socket.SocketKind,
                          Tuple[str, int]] = self._server_socket.accept()
            self.client_socket: socket.SocketKind = client[0]
            self.client_socket.settimeout(10)
        except Exception as error:
            print('Connection error: ', error)
            self._server_socket.close()
            sys.exit()

    def receive_data(self) -> bytes:
        """Receives data from the standalone application and forwards it to the caller."""
        data: bytes = b''
        try:
            message_length: int = int.from_bytes(self.client_socket.recv(4),
                                                 byteorder='little')

            while len(data) < message_length:
                chunk: bytes = self.client_socket.recv(
                    min(message_length - len(data), 2048))
                if chunk == b'':
                    raise RuntimeError('Socket connection broken.')
                data += chunk
        except socket.timeout as error:
            print('Timeout exception: ')
            print(error)
            return b''

        return data

    def start_unity(self) -> Tuple[Thread, Popen]:
        """Starts the Unity standalone."""
        thread: Thread = Thread(target=self._accept_connection)
        thread.start()
        sleep(1)

        process: Popen = start_standalone()

        self._current_process = process

        thread.join()

        def signal_handler(sig, frame):
            print('Closing the standalone.')
            self.close()
            exit()

        signal.signal(signal.SIGINT, signal_handler)

        return thread, process

    def start_new_game(self, map_info: CerealBarProto_pb2.MapInfo) -> None:
        """Starts a new game given a seed and the number of cards (which can deterministically set the
        props/terrain).
        """
        thread: Thread = Thread(target=self._accept_connection)
        thread.start()
        sleep(1)
        self.client_socket.send((f'restart').encode())

        thread.join()
        pb = map_info.SerializeToString()

        self.client_socket.send(b'start,' + pb)
        self.receive_data()

    def send_card_replacement(self, card_set: CerealBarProto_pb2.ScoreSetCard):
        pb = card_set.SerializeToString()

        self.client_socket.send(b'replacecards,' + pb)

        self.receive_data()

    def send_card_selection(
            self, card_selection: CerealBarProto_pb2.CardStatusChange) -> None:
        status_serialized = card_selection.SerializeToString()

        self.client_socket.send(b'cardchange,' + status_serialized)
        self.receive_data()

    def send_data(self, data: str) -> None:
        """Sends data to the Unity standalone."""
        self.client_socket.send(data.encode())

    def close(self):
        """Quits the standalone application."""
        self._server_socket.close()
        self.client_socket.close()
        quit_standalone(self._current_process)
