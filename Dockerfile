# Use specific version for stability
FROM python:3.10

# 1. Install System Dependencies (Poppler)
# We use apt-get because this image is based on Debian Linux
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code
COPY . .

# 5. Expose the port the app runs on
EXPOSE 8000

# 6. Command to start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]