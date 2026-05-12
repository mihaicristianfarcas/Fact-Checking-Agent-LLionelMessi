import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { WebSocketServer } from "ws";
import { networkInterfaces } from "os";

// Get local network IP
function getLocalIP(): string {
	const nets = networkInterfaces();
	for (const name of Object.keys(nets)) {
		for (const net of nets[name] || []) {
			if (net.family === "IPv4" && !net.internal) {
				return net.address;
			}
		}
	}
	return "localhost";
}

// Store for connected clients
const clients = new Set<import("ws").WebSocket>();
let currentSlide = 0;

// Must match TOTAL_SLIDES in src/App.tsx (14 slides for the fact-checking deck).
const MAX_SLIDE_INDEX = 13;

export default defineConfig({
	plugins: [
		react(),
		tailwindcss(),
		{
			name: "websocket-server",
			configureServer(server) {
				const wss = new WebSocketServer({ noServer: true });
				const localIP = getLocalIP();

				server.httpServer?.on("upgrade", (request, socket, head) => {
					if (request.url === "/ws") {
						wss.handleUpgrade(request, socket, head, (ws) => {
							wss.emit("connection", ws, request);
						});
					}
				});

				wss.on("connection", (ws) => {
					clients.add(ws);
					// Send current slide and IP to new connection
					ws.send(JSON.stringify({ type: "sync", slide: currentSlide, ip: localIP }));

					ws.on("message", (data) => {
						try {
							const msg = JSON.parse(data.toString());
							if (msg.type === "goto") {
								currentSlide = Math.max(0, Math.min(msg.slide, MAX_SLIDE_INDEX));
								// Broadcast to all clients
								clients.forEach((client) => {
									if (client.readyState === 1) {
										client.send(JSON.stringify({ type: "sync", slide: currentSlide }));
									}
								});
							} else if (msg.type === "next" || msg.type === "prev") {
								// Broadcast command to all clients
								clients.forEach((client) => {
									if (client.readyState === 1) {
										client.send(JSON.stringify({ type: msg.type }));
									}
								});
							}
						} catch (e) {
							console.error("WS parse error:", e);
						}
					});

					ws.on("close", () => {
						clients.delete(ws);
					});
				});
			},
		},
	],
	server: {
		host: true, // Enable network access
	},
});