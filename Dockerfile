# Use a more recent Python version to satisfy dependencies
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Create a non-root user to run the application
RUN useradd -m -u 1000 appuser
# Set an environment variable to tell sentence-transformers to use a local cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Run the preloading script as root to ensure permissions for downloading
RUN python preload_models.py

# Change the ownership of the code directory to the new user
RUN chown -R appuser:appuser /code

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (Gunicorn will use this)
EXPOSE 8080

# Define environment variable for the port
ENV PORT=8080

# Run the app with Gunicorn when the container launches
# --timeout 0 prevents the server from crashing during long model load times
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "0", "app:app"]
