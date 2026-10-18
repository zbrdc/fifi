#!/usr/bin/env python3
"""ACC-B rehearsal relay (design 9): the stand-in for `ssh -L` in --rehearse mode.

It listens on a unix socket and forwards every connection to one TCP target (the fake server;
the port comes from the launcher, which takes it from ACCB_TEST_PORT in a rehearsal), one thread
per connection, copying both directions and closing on error. Unlike the bridge it has no
connection cap: it stands in for the ssh forward,
which has none. SIGTERM (or SIGINT) ends it. The socket mode is fixed by the launcher, which
always runs `chmod 666` after the socket appears (design 4.2); this script relies on no umask.
Run with `python3 -I`.
"""

import argparse
import pathlib
import signal
import socket
import sys
import threading


def relay(client, host, port):
    try:
        upstream = socket.create_connection((host, port), timeout=10)
    except OSError:
        client.close()
        return

    def pump(source, sink):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        finally:
            try:
                sink.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    threads = [threading.Thread(target=pump, args=(client, upstream), daemon=True),
               threading.Thread(target=pump, args=(upstream, client), daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    client.close()
    upstream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--socket", required=True, help="the unix socket path to listen on")
    parser.add_argument("--target", required=True, help="HOST:PORT to forward to")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    host, _, port = args.target.partition(":")
    if not host or not port.isdigit():
        parser.error("--target must be HOST:PORT")
    path = pathlib.Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    stopping = threading.Event()

    def stop(*_):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(128)
    server.settimeout(0.5)
    try:
        while not stopping.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=relay, args=(client, host, int(port)), daemon=True).start()
    finally:
        server.close()
        try:
            path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
