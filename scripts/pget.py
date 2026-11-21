#!/usr/bin/env python3
"""Parallel ranged downloader (IPv4 only). Resumable: finished chunks are listed in <dest>.done.
  pget.py URL DEST [threads] [expected_size]
"""
import os, socket, sys, threading, urllib.request

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, *a, **k: _orig(h, p, socket.AF_INET, *a, **k)

url, dest = sys.argv[1], sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CH = 32 * 2**20


def size_of(u):
    for attempt in range(20):
        try:
            req = urllib.request.Request(u, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"]), r.geturl()
        except Exception as e:  # noqa: BLE001
            print(f"HEAD attempt {attempt}: {e!r}", flush=True)
            import time; time.sleep(5)
    sys.exit(f"HEAD failed: {u}")


size, final = size_of(url)
os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
fd = os.open(dest + ".part", os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, size)
done_path = dest + ".done"
done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
todo = [i for i in range((size + CH - 1) // CH) if i not in done]
lock = threading.Lock()
print(f"{dest}: {size} bytes, {len(todo)} chunks left", flush=True)


def worker():
    while True:
        with lock:
            if not todo:
                return
            i = todo.pop(0)
        lo, hi = i * CH, min(size, (i + 1) * CH) - 1
        for attempt in range(30):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != hi - lo + 1:
                    raise IOError("short read")
                os.pwrite(fd, data, lo)
                with lock:
                    with open(done_path, "a") as f:
                        f.write(f"{i}\n")
                break
            except Exception as e:  # noqa: BLE001
                print(f"chunk {i} attempt {attempt}: {e!r}", flush=True)
                import time; time.sleep(3)
        else:
            print(f"chunk {i} FAILED", flush=True)


ts = [threading.Thread(target=worker) for _ in range(threads)]
[t.start() for t in ts]
[t.join() for t in ts]
os.close(fd)
n_done = len(set(open(done_path).read().split()))
if n_done == (size + CH - 1) // CH:
    os.replace(dest + ".part", dest)
    os.remove(done_path)
    print("COMPLETE", dest, flush=True)
else:
    print("INCOMPLETE", dest, n_done, flush=True)
    sys.exit(1)
