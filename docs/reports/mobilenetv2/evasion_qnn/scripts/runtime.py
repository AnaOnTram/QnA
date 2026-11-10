"""Host adapter to the hash-pinned persistent Q6A native-input QNN service."""
import hashlib
import http.client
import json
from pathlib import Path
import socket
import struct
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT.parent
HOST='192.168.1.40'
PORT=8088


class Runtime:
    def __init__(self):
        self.connection=http.client.HTTPConnection(HOST,PORT,timeout=120)
        self.connection.connect()
        self.connection.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.calls=0;self.queries=0;self.seconds=0.
        self.info=self.health()
        expected=hashlib.sha256((MODEL/'qnn/mobilenetv2_quantized.bin').read_bytes()).hexdigest()
        assert self.info['ready'] and self.info['context_sha256']==expected
        assert self.info['input']['shape']==[1,3,224,224]
        assert self.info['output']['shape']==[1,1000]
        self.mean=np.array([.485,.456,.406],np.float32)[None,:,None,None]
        self.std=np.array([.229,.224,.225],np.float32)[None,:,None,None]

    def health(self):
        self.connection.request('GET','/health')
        response=self.connection.getresponse();body=response.read()
        assert response.status==200,body
        return json.loads(body)

    def encode(self,pixels):
        x=pixels.detach().cpu().numpy().astype('float32')
        x=(x-self.mean)/self.std
        return np.clip(np.rint(x/np.float32(self.info['input']['scale']))-self.info['input']['offset'],0,255).astype('uint8')

    @torch.no_grad()
    def predict(self,pixels):
        encoded=self.encode(pixels)
        n=len(encoded)
        payload=struct.pack('<I',n)+encoded.tobytes()
        t=time.perf_counter()
        self.connection.request('POST','/predict',body=payload,headers={'Content-Type':'application/octet-stream'})
        response=self.connection.getresponse();body=response.read()
        assert response.status==200,(response.status,body[:200])
        assert len(body)==4+n*1000 and struct.unpack('<I',body[:4])[0]==n
        self.calls+=1;self.queries+=n;self.seconds+=time.perf_counter()-t
        native=np.frombuffer(body[4:],dtype='uint8').reshape(n,1000).astype('float32')
        output=(native+self.info['output']['offset'])*np.float32(self.info['output']['scale'])
        return torch.from_numpy(output)

    def stats(self):
        return dict(calls=self.calls,queries=self.queries,roundtrip_seconds=self.seconds)

    def close(self):self.connection.close()
