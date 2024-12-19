import os
import requests
import whisper
import subprocess
from pydub import AudioSegment
import ssl
import time
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

class TranscriptionRequest(BaseModel):
    mp3_url: str

def download_file(url, save_path):
    """Скачивает mp3-файл по ссылке."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Файл скачан: {save_path}")
    else:
        raise Exception(f"Ошибка при скачивании файла: {response.status_code}")


def convert_to_wav(mp3_path, wav_path, sample_rate=16000):
    """Конвертирует mp3 в WAV с частотой дискретизации 16 kHz."""
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(2)
    audio.export(wav_path, format="wav")
    print(f"Файл конвертирован в WAV: {wav_path}")


def split_channels(wav_path, client_path, manager_path):
    """
    Разделяет стерео WAV на два монофайла.
    Левый канал = Клиент, Правый канал = Менеджер
    """
    # Левый канал (1) -> Клиент
    subprocess.run(["sox", wav_path, client_path, "remix", "1"], check=True)
    # Правый канал (2) -> Менеджер
    subprocess.run(["sox", wav_path, manager_path, "remix", "2"], check=True)
    print(f"Каналы разделены:\n  Клиент: {client_path}\n  Менеджер: {manager_path}")


def normalize_audio(input_path, output_path):
    """Нормализует аудио с помощью sox."""
    try:
        subprocess.run(
            [
                "sox", input_path, "-r", "16k", output_path,
                "norm", "-0.5",
                "compand", "0.3,1", "-90,-90,-70,-70,-60,-20,0,0", "-5", "0", "0.2"
            ],
            check=True
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при нормализации аудио: {e}")


def transcribe_with_timestamps(audio_path, model):
    """Распознаёт текст с тайм-кодами."""
    result = model.transcribe(audio_path, language="ru", task="transcribe")
    return result["segments"]


def format_dialogue(segments_manager, segments_client):
    """Форматирует текст в виде ролевки с тайм-кодами, упорядочивая по времени."""
    # Объединяем все сегменты в один список, добавляя роль
    tagged_segments = []
    for seg in segments_manager:
        tagged_segments.append((seg['start'], seg['end'], 'Менеджер', seg['text']))
    for seg in segments_client:
        tagged_segments.append((seg['start'], seg['end'], 'Клиент', seg['text']))

    # Сортируем по времени начала
    tagged_segments.sort(key=lambda x: x[0])

    # Формируем итоговую строку
    dialogue_lines = []
    for start, end, role, text in tagged_segments:
        dialogue_lines.append(f"[{start:.2f}-{end:.2f}] {role}: {text.strip()}")

    return "\n".join(dialogue_lines)


def main(mp3_url):
    start_time = time.time()  # Начало измерения времени

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Параметры
    mp3_path = "input.mp3"
    wav_path = "input.wav"
    client_wav = "client.wav"
    manager_wav = "manager.wav"
    left_normalized = os.path.join(script_dir, "record-normalized-left.wav")
    right_normalized = os.path.join(script_dir, "record-normalized-right.wav")

    try:
        # Шаг 1: Скачиваем MP3
        download_file(mp3_url, mp3_path)

        # Шаг 2: Конвертируем в WAV (16 kHz)
        convert_to_wav(mp3_path, wav_path)

        # Шаг 3: Разделяем каналы (левый = Клиент, правый = Менеджер)
        split_channels(wav_path, client_wav, manager_wav)

        # Нормализация аудио
        normalize_audio(client_wav, left_normalized)
        normalize_audio(manager_wav, right_normalized)

        # Шаг 4: Распознаём текст
        model = whisper.load_model("large")

        print("Распознавание текста для менеджера...")
        segments_manager = transcribe_with_timestamps(right_normalized, model)

        print("Распознавание текста для клиента...")
        segments_client = transcribe_with_timestamps(left_normalized, model)

        # Шаг 5: Форматируем диалог
        dialogue = format_dialogue(segments_manager, segments_client)

        # Шаг 6: Сохранение результата (опционально)
        with open("dialogue.txt", "w", encoding="utf-8") as f:
            f.write(dialogue)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        dialogue = ""
        execution_time = 0.0
        return dialogue, execution_time

    end_time = time.time()  # Конец измерения времени
    execution_time = end_time - start_time

    return dialogue, execution_time


@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    # Поскольку main - синхронная функция, используем run_in_threadpool для асинхронного вызова
    try:
        dialogue, exec_time = await run_in_threadpool(main, request.mp3_url)
        return {
            "dialogue": dialogue,
            "execution_time": exec_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}
