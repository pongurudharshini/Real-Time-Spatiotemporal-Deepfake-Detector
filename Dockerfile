# 1. Start with a lightweight Python 3.12 server environment
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install low-level system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your actual code into the container
COPY . .

# 6. Expose the port Streamlit uses
EXPOSE 8501

# 7. Command to boot the dashboard when the server turns on
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]