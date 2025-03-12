FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy backend dependencies (adjust the file name/path if needed)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete backend code
COPY . .

# Expose the port used by Flask (as defined in main.py, e.g. 8000)
EXPOSE 8000

# Run the Flask app
CMD ["python", "main.py"]