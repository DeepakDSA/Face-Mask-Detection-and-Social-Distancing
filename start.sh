#!/bin/sh
set -e
# This command starts the Gunicorn server, which will run your app.py
# It binds to the port specified by the PORT environment variable from Render.
gunicorn --workers=2 --threads=4 --timeout 120 --bind 0.0.0.0:$PORT app:app
