"""
Tiny local server so the browser will grant camera access
(getUserMedia requires http://localhost, not file://).

Run:  python serve.py
Then open the printed URL in Chrome or Edge.
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}/index.html"
    print(f"Serving RehabSense AI web UI at {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    httpd.serve_forever()
