# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes app.py, utils.py, the templates folder, and the models folder
COPY . .

# Make port 80 available to the world outside this container
# Hosting services like Render will use this port.
EXPOSE 80

# Define environment variable for the port
ENV PORT 80

# Run app.py when the container launches using Gunicorn
# We bind to 0.0.0.0 to make it accessible from outside the container.
# The number of workers can be tuned for performance.
CMD ["gunicorn", "--workers=2", "--threads=4", "--bind=0.0.0.0:$PORT", "app:app"]
