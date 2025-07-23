# Use a more recent Python version to satisfy dependencies
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# --- FIX FOR PERMISSION ERROR ---
# Set an environment variable to tell sentence-transformers to use a local cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache

# Create the cache directory and give it the correct permissions
RUN mkdir -p /code/cache && chmod -R 777 /code/cache

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable for the port (Hugging Face will set this)
ENV PORT=8080

# Run app.py when the container launches
CMD ["python", "app.py"]
# Use the gunicorn server for better performance in production