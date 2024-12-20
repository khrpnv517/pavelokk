# Используем базовый образ с поддержкой CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем символическую ссылку для python3 и pip3
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем pip
RUN pip install --upgrade pip

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .

# Устанавливаем зависимости Python с поддержкой CUDA
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Указываем переменную окружения для корректного вывода логов
ENV PYTHONUNBUFFERED=1

# Запуск приложения с использованием Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]