// ACC-B in-container bridge (design 4.4): 127.0.0.1:18791 <-> /ipc/fifi.sock, node `net` only.
// It adds no header and reads no key. The bearer key is readable by every process in the
// container by design (user approval U4: the key is random, valid for one server run, and
// authorizes only this server and run); the bridge never needs it. At most
// MAX_CONNECTIONS connections are served at a time; further connections are closed at once
// (design C13). The model can kill or replace this process; that gains nothing but availability.
// At startup it connects once to the relay socket and exits if that fails, so the entry script
// can stop with status 65 when the forward is missing (design 4.4 step 3, check E4).
"use strict";
const net = require("net");
const LISTEN_HOST = "127.0.0.1";
const LISTEN_PORT = 18791;
const SOCKET_PATH = "/ipc/fifi.sock";
const MAX_CONNECTIONS = 8;

let probeOk = false;
const probe = net.connect(SOCKET_PATH);
probe.on("connect", () => { probeOk = true; probe.destroy(); });
probe.on("error", () => { if (!probeOk) process.exit(1); });

let open = 0;
const server = net.createServer((client) => {
  if (open >= MAX_CONNECTIONS) { client.destroy(); return; }
  open += 1;
  const upstream = net.connect(SOCKET_PATH);
  const done = () => { client.destroy(); upstream.destroy(); };
  client.on("error", done);
  upstream.on("error", done);
  client.on("close", () => { upstream.destroy(); open -= 1; });
  upstream.on("close", () => client.destroy());
  client.pipe(upstream);
  upstream.pipe(client);
});
server.listen(LISTEN_PORT, LISTEN_HOST);
