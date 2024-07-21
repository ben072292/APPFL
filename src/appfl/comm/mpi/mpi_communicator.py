import pickle
import numpy as np
from mpi4py import MPI
from collections import OrderedDict
from appfl.compressor import Compressor
from typing import Any, Optional, Union
import logging
import time

# Create a custom logger
logger = logging.getLogger('mpi_communicator_log')

# Set the log level
logger.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handler
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


class MpiCommunicator:
    """
    A general MPI communicator for synchronous or asynchronous distributed/federated/decentralized 
    learning experiments on multiple MPI processes, where each MPI process represents EXACTLY ONE learning client.
    """
    def __init__(self, comm: MPI.Intracomm, compresser: Optional[Compressor]=None):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.dests = []
        self.queue_status = [False for _ in range(self.comm_size - 1)]
        self.compressor = compresser

    def _obj_to_bytes(self, obj: Any) -> bytes:
        """Convert an object to bytes."""
        return pickle.dumps(obj)
    
    def _bytes_to_obj(self, bytes_obj: bytes) -> Any:
        """Convert bytes to an object."""
        return pickle.loads(bytes_obj)

    def scatter(self, contents, source: int) -> Any:
        """
        Scattering the `contents` to all MPI processes from the `source`
        :param `contents`: a list/sequence of contents to be scattered to all MPI processes
        :param `source`: the rank of the source MPI process that scatters the contents
        :return `content`: the content received by the current MPI process
        """
        if source == self.comm_rank:
            assert (
                len(contents) == self.comm_size
            ), "The size of the contents is not equal to the number of clients in scatter!"
        content = self.comm.scatter(contents, root=source)
        return content

    def gather(self, content, dest: int):
        """
        Gathering contents from all MPI processes to the destination process
        :param `content`: the content to be gathered from all MPI processes
        :param `dest`: the rank of the destination MPI process that gathers the contents
        :return `contents`: a list of contents received by the destination MPI process
        """
        contents = self.comm.gather(content, root=dest)
        return contents

    def broadcast_global_model(self, model: Optional[Union[dict, OrderedDict]]=None, args: Optional[dict]=None):
        """
        Broadcast the global model state dictionary and additional arguments from the server MPI process 
        to all other client processes. The method is ONLY called by the server MPI process.
        :param `model`: the global model state dictionary to be broadcasted
        :param `args`: additional arguments to be broadcasted
        """
        self.dests = (
            [i for i in range(self.comm_size) if i != self.comm_rank]
            if len(self.dests) == 0
            else self.dests
        )
        model_size = np.array([-1], dtype=np.int32)
        args_size = np.array([-1], dtype=np.int32)
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            for i in self.dests:
                serialized_args = self._obj_to_bytes(args)
                model_size[0] = 0
                args_size[0] = len(serialized_args)
                self.comm.Send([model_size, MPI.INT], dest=i, tag=i)
                self.comm.Send([args_size, MPI.INT], dest=i, tag=i + self.comm_size)
                self.comm.Send([bytearray(serialized_args), MPI.BYTE], dest=i, tag=i + self.comm_size * 2)
        else: # model is not None
            for i in self.dests:
                model_bytes = self._obj_to_bytes(model)
                model_size[0] = len(model_bytes)
                self.comm.Send([model_size, MPI.INT], dest=i, tag=i)
                if args is not None:
                    serialized_args = self._obj_to_bytes(args)
                    args_size[0] = len(serialized_args)
                    self.comm.Send([args_size, MPI.INT], dest=i, tag=i + self.comm_size)
                    self.comm.Send([bytearray(serialized_args), MPI.BYTE], dest=i, tag=i + self.comm_size * 2)
                else:
                    args_size[0] = 0
                    self.comm.Send([args_size, MPI.INT], dest=i, tag=i + self.comm_size)

                broadcast_time = time.time()
                self.comm.Send(
                    np.frombuffer(model_bytes, dtype=np.byte),
                    dest=i,
                    tag=i + self.comm_size * 3
                )
                logger.info("Step 1.%04d - MPI_Send: Server broadcasted global model (%d bytes) to client %d, took %.4f seconds", i-1, len(model_bytes), i-1, time.time() - broadcast_time)
            self.queue_status = [True for _ in range(self.comm_size - 1)]

    def send_global_model_to_client(self, model: Optional[Union[dict, OrderedDict]]=None, args: Optional[dict]=None, client_idx: int=-1):
        """
        Send the global model state dict and additional arguments to a certain client
        :param `model`: the global model state dictionary to be sent
        :param `args`: additional arguments to be sent
        :param `client_idx`: the index of the destination client
        """
        assert (
            client_idx >= 0 and client_idx < self.comm_size
        ), "Please provide a valid destination client index!"

        self.dests = (
            [i for i in range(self.comm_size) if i != self.comm_rank]
            if len(self.dests) == 0
            else self.dests
        )
        model_size = np.array([-1], dtype=np.int32)
        args_size = np.array([-1], dtype=np.int32)
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            serialized_args = self._obj_to_bytes(args)
            model_size[0] = 0
            args_size[0] = len(serialized_args)
            self.comm.Send([model_size, MPI.INT], dest=self.dests[client_idx], tag=self.dests[client_idx])
            self.comm.Send([args_size, MPI.INT], dest=self.dests[client_idx], tag=self.dests[client_idx] + self.comm_size)
            self.comm.Send([bytearray(serialized_args), MPI.BYTE], dest=self.dests[client_idx], tag=self.dests[client_idx] + self.comm_size * 2)
        else: # model is not None
            model_bytes = self._obj_to_bytes(model)
            model_size[0] = len(model_bytes)
            self.comm.Send([model_size, MPI.INT], dest=self.dests[client_idx], tag=self.dests[client_idx])
            if args is not None:
                serialized_args = self._obj_to_bytes(args)
                args_size[0] = len(serialized_args)
                self.comm.Send([args_size, MPI.INT], dest=self.dests[client_idx], tag=self.dests[client_idx] + self.comm_size)
                self.comm.Send([bytearray(serialized_args), MPI.BYTE], dest=self.dests[client_idx], tag=self.dests[client_idx] + self.comm_size * 2)
            else:
                args_size[0] = 0
                self.comm.Send([args_size, MPI.INT], dest=self.dests[client_idx], tag=self.dests[client_idx] + self.comm_size)

            send_time = time.time()
            self.comm.Send(
                np.frombuffer(model_bytes, dtype=np.byte),
                dest=self.dests[client_idx],
                tag=self.dests[client_idx] + self.comm_size * 3
            )
            logger.info("Step 1.%04d - MPI_Send: Server sent global model (%d bytes) to client %d, took %.4f seconds", client_idx, len(model_bytes), client_idx, time.time() - send_time)
            self.queue_status[client_idx] = True

    def send_local_model_to_server(self, model: Union[dict, OrderedDict], dest: int):
        """
        Client sends the local model state dict to the server (dest).
        :param `model`: the local model state dictionary to be sent
        :param `dest`: the rank of the destination MPI process (server)
        """
        if self.compressor is not None:
            model_bytes, _ = self.compressor.compress_model(model)
        else:
            model_bytes = self._obj_to_bytes(model)
        # logger.info("Step 3.%04d - MPI_Send: Client %d queued local model size (%d) to server", self.comm_rank-1, self.comm_rank-1, len(buffer))
        send_time = time.time()
        self.comm.Send(
            np.frombuffer(model_bytes, dtype=np.byte),
            dest=dest,
            tag=self.comm_rank + self.comm_size * 3,
        )
        logger.info("Step 3.%04d - MPI_Send: Client %d sent local model data (%d bytes) to server, took %.4f seconds", self.comm_rank-1, self.comm_rank-1, len(model_bytes), time.time() - send_time)

    def recv_local_model_from_client(self, model_copy=None):
        """
        Server receives the local model state dict from one finishing client.
        :param `model_copy` [Optional]: a copy of the global model state dict ONLY used for decompression
        """
        status = MPI.Status()
        while True:
            time.sleep(0.1)
            msg_flag = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if msg_flag:
                source = status.Get_source()
                client_idx = source -1
                tag = status.Get_tag()
                count = status.Get_count(MPI.BYTE)
                model_bytes = np.zeros(count, dtype=np.byte)
                # logger.info("Step 4.%04d - MPI_Recv: Server receiving client %d's model (%d bytes)", client_idx, client_idx, len(model_bytes))
                recv_time = time.time()
                self.comm.Recv(
                    model_bytes,
                    source=self.dests[client_idx],
                    tag=self.dests[client_idx] + self.comm_size * 3,
                )
                logger.info("Step 4.%04d - MPI_Recv: Server received client %d's model (%d bytes), took %.4f seconds", client_idx, client_idx, len(model_bytes), time.time() - recv_time)
                if self.compressor is not None:
                    model = self.compressor.decompress_model(model_bytes, model_copy)
                else:
                    model = self._bytes_to_obj(model_bytes.tobytes())
                self.queue_status[client_idx] = False
                return client_idx, model

    def recv_global_model_from_server(self, source):
        """
        Client receives the global model state dict from the server (source).
        :param `source`: the rank of the source MPI process (server)
        :return (`model`, `args`): the global model state dictionary and optional arguments received by the client
        """
        ################################ Implicit MPI Recv does not work on NCSA Delta
        # meta_data = self.comm.recv(source=source, tag=self.comm_rank)
        # if isinstance(meta_data, tuple):
        #     model_size, args = meta_data[0], meta_data[1]
        # else:
        #     model_size, args = meta_data, None
        # if model_size == 0:
        #     return None, args
        # model_bytes = np.empty(model_size, dtype=np.byte)

        # self.comm.Recv(model_bytes, source=source, tag=self.comm_rank + self.comm_size)
        # model = self._bytes_to_obj(model_bytes.tobytes())
        # return model if args is None else (model, args)

        ################################ My modified code to use explicit buffer
        model_size_buffer = np.empty(1, dtype=np.int32)
        args_size_buffer = np.empty(1, dtype=np.int32)

        # Receive meta_data into the buffer
        # logger.info("Client %d (Step 1) - MPI_Recv:  Receiving metadata from server", self.comm_rank)
        self.comm.Recv(model_size_buffer, source=source, tag=self.comm_rank)
        self.comm.Recv(args_size_buffer, source=source, tag=self.comm_rank + self.comm_size)
        # Extract the received values from the buffers
        model_size = model_size_buffer[0]
        args_size = args_size_buffer[0]
        if args_size > 0:
            args_buffer = np.empty(args_size, dtype=np.byte)
            self.comm.Recv(args_buffer, source=source, tag=self.comm_rank + self.comm_size * 2)
            args = self._bytes_to_obj(args_buffer.tobytes())
        else:
            args = None
        if model_size > 0:
            model_bytes = np.empty(model_size, dtype=np.byte)
            recv_time = time.time()
            logger.info("Step 2.%04d - MPI_Recv: Client %d receiving global model from sever (%d bytes), took %.4f seconds", self.comm_rank-1, self.comm_rank-1, model_bytes.itemsize*model_bytes.size, time.time() - recv_time)
            self.comm.Recv(model_bytes, source=source, tag=self.comm_rank + self.comm_size * 3)
            model = self._bytes_to_obj(model_bytes.tobytes())
        else:
            model = None
        return model if args is None else (model, args)

    def cleanup(self):
        """
        Clean up the MPI communicator by waiting for all pending requests.
        """
        status = MPI.Status()
        while sum(self.queue_status) > 0:
            time.sleep(0.1)
            msg_flag = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if msg_flag:
                source = status.Get_source()
                client_idx = source -1
                tag = status.Get_tag()
                count = status.Get_count(MPI.BYTE)
                model_bytes = np.zeros(count, dtype=np.byte)
                self.comm.Recv(
                    model_bytes,
                    source=self.dests[client_idx],
                    tag=self.dests[client_idx] + self.comm_size * 3,
                )
                logger.info("Step 5.%04d - MPI_Recv: Server received model from client %d (%d bytes) before cleanup. Waiting for other %d clients", self.dests[client_idx]-1, self.dests[client_idx]-1, model_bytes.itemsize*model_bytes.size, sum(self.queue_status)-1)
                self.queue_status[client_idx] = False
