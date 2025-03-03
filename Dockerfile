# Use the official Python image as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user with a home directory
RUN useradd -m -u 1000 appuser

# Set environment variables for cache directories
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers
ENV TORCH_HOME=/home/appuser/.cache/torch

# Create cache directories and set permissions
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $TORCH_HOME \
    && chown -R appuser:appuser /home/appuser/.cache

# Switch to the non-root user
USER appuser

# Copy the rest of the application code
COPY . .

# Set the environment variable for the Hugging Face API token
ENV HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
