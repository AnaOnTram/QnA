"""Persistent LAN QNN service. POST /predict: uint32 count + native UINT8 inputs."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
lock = threading.Lock()
scorer = None
calls = 0
queries = 0
started = time.time()


def readexact(stream, n):
    out = bytearray()
    while len(out) < n:
        part = stream.read(n-len(out))
        if not part:
            raise EOFError('scorer stopped')
        out.extend(part)
    return bytes(out)


def memory(pid):
    try:
        return {k: int(v.split()[0]) for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines()) if k in ('VmRSS','VmHWM')}
    except OSError:
        return {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

    def reply(self, status, body, kind='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404,b'{}')
        value = dict(ready=scorer.poll() is None, context_sha256=CONTEXT_SHA,
                     graph='mobilenetv2_fp32', runtime='QAIRT 2.42 HTP v68',
                     pid=os.getpid(), scorer_pid=scorer.pid, calls=calls, queries=queries,
                     uptime_seconds=time.time()-started, server_memory_kib=memory(os.getpid()),
                     scorer_memory_kib=memory(scorer.pid),
                     input=dict(dtype='uint8',shape=[1,3,224,224],scale=.01865844801068306,offset=-114),
                     output=dict(dtype='uint8',shape=[1,1000],scale=.14984077215194702,offset=-71))
        self.reply(200,json.dumps(value).encode())

    def do_POST(self):
        global calls, queries
        if self.path != '/predict':
            return self.reply(404,b'{}')
        length = int(self.headers.get('Content-Length','0'))
        if length < 4 or length > 4+100*150528:
            return self.reply(400,b'{"error":"invalid length"}')
        body = self.rfile.read(length)
        n = struct.unpack('<I',body[:4])[0]
        if n < 1 or n > 100 or length != 4+n*150528:
            return self.reply(400,b'{"error":"invalid batch"}')
        try:
            # Interleave input/output: the scorer streams each sample and cannot
            # consume the entire batch while its stdout pipe is full.
            with lock:
                scorer.stdin.write(body[:4])
                outputs = bytearray()
                for i in range(n):
                    scorer.stdin.write(body[4+i*150528:4+(i+1)*150528])
                    scorer.stdin.flush()
                    if i == 0:
                        count = readexact(scorer.stdout,4)
                        assert struct.unpack('<I',count)[0] == n
                        outputs.extend(count)
                    outputs.extend(readexact(scorer.stdout,1000))
                calls += 1
                queries += n
            self.reply(200,bytes(outputs),'application/octet-stream')
        except Exception as exc:
            self.reply(503,json.dumps(dict(error=str(exc))).encode())

    def log_message(self, fmt, *args):
        if self.path != '/predict':
            super().log_message(fmt,*args)


if __name__ == '__main__':
    CONTEXT_SHA = hashlib.sha256(Path('mobilenetv2_quantized.bin').read_bytes()).hexdigest()
    env = dict(os.environ,LD_LIBRARY_PATH=str(ROOT),ADSP_LIBRARY_PATH=str(ROOT),CDSP_LIBRARY_PATH=str(ROOT))
    with open('service_scorer.log','ab') as log:
        scorer = subprocess.Popen(['./qnn-scorer','mobilenetv2_quantized.bin','mobilenetv2_fp32','libQnnSystem.so','1','367'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0,env=env)
        assert scorer.stdout.readline().startswith(b'READY 150528 1000 ')
        def stop(*_):
            scorer.stdin.close()
            scorer.wait(timeout=15)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM,stop)
        try:
            print('Serving MobileNetV2 QNN HTP on port 8088',flush=True)
            ThreadingHTTPServer(('0.0.0.0',8088),Handler).serve_forever()
        finally:
            if scorer.poll() is None:
                scorer.stdin.close()
                scorer.wait(timeout=15)
