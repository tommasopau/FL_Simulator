FROM python:3.10-slim


RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire backend code
COPY . .

# Expose port 8000 (Flask app)
EXPOSE 8000

# Run the app; main_bp.py is the entrypoint for the Flask app
CMD ["python", "main_bp.py"]