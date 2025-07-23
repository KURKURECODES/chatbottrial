# Use a more recent Python version to satisfy dependencies
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# --- FIX FOR PERMISSION ERROR ---
# Create a non-root user to run the application
RUN useradd -m -u 1000 appuser
# Set an environment variable to tell sentence-transformers to use a local cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Change the ownership of the code directory to the new user
RUN chown -R appuser:appuser /code

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable for the port (Hugging Face will set this)
ENV PORT=8080

# Run app.py when the container launches
CMD ["python", "app.py"]
