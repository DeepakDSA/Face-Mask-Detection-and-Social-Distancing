# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port available to the world outside this container
# Render will automatically use the correct port.
EXPOSE 10000

# Run app.py when the container launches using Gunicorn
# This is the corrected line that fixes the $PORT error.
CMD gunicorn --workers=2 --threads=4 --bind 0.0.0.0:$PORT app:app
