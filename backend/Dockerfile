FROM python:3.9-slim

WORKDIR /app

# Copy backend dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete backend code
COPY . .

EXPOSE 8000

# Launch the backend using main_bp.py as per README instructions
CMD ["python", "main_bp.py"]