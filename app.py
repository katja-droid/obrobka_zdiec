import time
import numpy as np
import cv2
import psutil
import tracemalloc
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
from numba import jit, prange
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

app = FastAPI()

# Serve static files (CSS, JS, images)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up HTML template rendering
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "processed_image": None, "profiling_chart": None})

# Metoda OpenCV
def opencv_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Metoda NumPy + Numba z SIMD i wielowątkowością
@jit(nopython=True, parallel=True)
def numba_grayscale(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    
    for i in prange(height):
        for j in prange(width):
            r, g, b = image[i, j]
            gray[i, j] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    
    return gray

# Funkcja do testowania wydajności i zużycia pamięci
def benchmark(func, img, runs=10):
    times = []
    tracemalloc.start()
    
    for _ in range(runs):
        start_mem = psutil.Process().memory_info().rss  # Pamięć przed
        start_time = time.time()
        _ = func(img)
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss  # Pamięć po
        times.append((end_time - start_time) * 1000)  # Konwersja na ms
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    mem_usage = (peak_mem - current_mem) / 1e6  # Konwersja na MB
    return avg_time, std_time, mem_usage

# Wykres profilowania funkcji
def plot_profiling(opencv_time, numba_time):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(["OpenCV", "Numba"], [opencv_time, numba_time], color=['blue', 'red'])
    for bar, time in zip(bars, [opencv_time, numba_time]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{time:.2f} ms", va='center', fontsize=12)
    ax.set_xlabel("Czas (ms)")
    ax.set_title("Porównanie SIMD i wielowątkowości")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.savefig("static/profiling.png")
    plt.close()

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid image", "processed_image": None, "profiling_chart": None})
    
    # Testowanie metod
    opencv_time, _, _ = benchmark(opencv_grayscale, img)
    numba_time, _, _ = benchmark(numba_grayscale, img)
    plot_profiling(opencv_time, numba_time)
    
    # Przetwarzanie obrazu i zapis
    processed_img = opencv_grayscale(img)
    processed_image_path = "static/processed_image.jpg"
    cv2.imwrite(processed_image_path, processed_img)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "processed_image": "/" + processed_image_path,
        "profiling_chart": "/static/profiling.png"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
