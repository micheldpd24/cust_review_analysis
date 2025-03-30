# Base Image : Python 3.11 slim
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Dependencies for matplotlib and wordcloud
RUN apt-get update && apt-get install -y \
   gcc \
   g++ \
   libfreetype6-dev \
   libpng-dev \
   pkg-config \
   && rm -rf /var/lib/apt/lists/*

# Copy local file into the container 
COPY requirements.txt .
COPY app.py .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8050
EXPOSE 8050

# Commande pour lancer l'application
CMD ["python", "app.py"]