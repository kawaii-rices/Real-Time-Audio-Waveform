# Real-Time-Audio-Waveform
import sys # to have clean exit of loops
import pygame # it is for window, drawing, text, events
import numpy as np # fast math on arrays(RMS,scaling)
import pyaudio # for microphone input 
# ----------------------- Config -----------------------
# PYGAME SETUP :-
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
CENTER_Y = SCREEN_HEIGHT // 2
WAVE_COLOR = (255,255,255)
BG_COLOR = (10,20,40)
TEXT_COLOR = (255,255,0)
LINE_THICKNESS = 2

# dB display settings
DB_FLOOR = -60.0

# window + caption
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Real-time Audio Waveform")
# font + clock(60 fps)
font = pygame.font.Font(None, 32)
clock = pygame.time.Clock() 
FPS = 60

# ----------------------- Slider (for EMA smoothing) -----------------------
# This slider controls 'alpha' in the EMA update:
# ema_db = (1 - alpha) * ema_db + alpha * inst_db
# Smaller alpha -> slower, smoother meter. Larger alpha -> faster, more twitchy.
ALPHA_MIN = 0.05
ALPHA_MAX = 0.60
alpha = 0.20          # starting smoothing
ema_db = DB_FLOOR     # start “quiet”

# Slider UI geometry (bottom-left)
SLIDER_W = 260
SLIDER_H = 10
SLIDER_X = 16
SLIDER_Y = SCREEN_HEIGHT - 28
HANDLE_W = 12
dragging = False  # track if user is dragging the slider handle

def alpha_to_x(a):
    """Map alpha in [ALPHA_MIN..ALPHA_MAX] to handle x pixel."""
    t = (a - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
    return SLIDER_X + int(t * SLIDER_W)

def x_to_alpha(x):
    """Map handle x pixel back to alpha in [ALPHA_MIN..ALPHA_MAX]."""
    t = (x - SLIDER_X) / float(SLIDER_W)
    t = max(0.0, min(1.0, t))
    return ALPHA_MIN + t * (ALPHA_MAX - ALPHA_MIN)

# --------------------------------------------------------------------------

# PYAUDIO (microphone) SETUP:-

FORMAT = pyaudio.paInt16   # 16-bit integers format
CHANNELS = 1               # monoaudio (only one no left or right)
RATE = 44100               # 44.1 kHz same as mp4 and youtube etc.
CHUNK = SCREEN_WIDTH       # 1 sample = 1 pixel

pa = pyaudio.PyAudio() # Connecting it all

stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

# format=FORMAT
# It tells PyAudio:
# “Give me samples as 16-bit integers.”
# channels=CHANNELS
# It tells PyAudio:
# “I only need microphone input in mono.”
# rate=RATE
# It tells PyAudio:
# “The microphone should send data at 44,100 samples per second.”
# input=True
# It are recording, not playing sound.
# frames_per_buffer=CHUNK
# It tells PyAudio:
# “Every time I call read(), give me EXACTLY this many samples.”

# MAIN LOOPS : - 
running = True
while running:
    # events (pygame)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

        # ---------------- Slider mouse events ----------------
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # handle rectangle
            handle_x = alpha_to_x(alpha) - HANDLE_W // 2
            handle_rect = pygame.Rect(handle_x, SLIDER_Y - (SLIDER_H // 2), HANDLE_W, SLIDER_H * 3)
            track_rect = pygame.Rect(SLIDER_X, SLIDER_Y - SLIDER_H // 2, SLIDER_W, SLIDER_H * 3)
            if handle_rect.collidepoint(mx, my) or track_rect.collidepoint(mx, my):
                dragging = True
                alpha = x_to_alpha(mx)
        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        if event.type == pygame.MOUSEMOTION and dragging:
            mx, _ = event.pos
            alpha = x_to_alpha(mx)
        # -----------------------------------------------------

    # read audio (numpy)
    data = stream.read(CHUNK, exception_on_overflow=False)
    # exception_on_overflow=False says: don’t raise an error; just give me whatever is there (best-effort).
    # This avoids crashes but you might see a tiny visual “jump” if it happens.
    samples_i16 = np.frombuffer(data, dtype=np.int16)
    # np.frombuffer creates a NumPy array view over existing memory:
    # Zero-copy (super fast).
    # Shape: (CHUNK,) → here (800,).
    # Values are in the range −32768 … 32767 (standard 16-bit PCM).
    samples = samples_i16.astype(np.float32) / 32768.0 
    # np.float32 converts arrays from int 16 --> float32 cuz they are faster

    # This normalizes samples to the standard audio float range ~ [-1.0, +1.0).
    # Why 32768 and not 32767?
    # int16 range is asymmetric: min = -32768, max = +32767.
    # Dividing by 32768 maps:
    # -32768 → -1.0 exactly
    # +32767 → +0.99997… (just under +1.0)
    # This keeps the negative and positive ranges balanced and avoids ever producing a float > 1.0.
    # Alternative you’ll see: / 32767.0. That makes +32767 → +1.0, but -32768 → -1.00003… (slightly < -1), so you’d need to clip. Using 32768 avoids that headache.

    # Compute loudness (RMS --> dBFS)

    rms = np.sqrt(np.mean(samples**2))     # RMS of normalized signal
    inst_db = -60.0 if rms < 1e-6 else 20.0 * np.log10(rms)  # dBFS (0 dBFS = max)
    # EMA smoothing controlled by slider 'alpha'
    ema_db = (1 - alpha) * ema_db + alpha * inst_db

    # Make waveform points (for drawing)
    ys = (samples * (SCREEN_HEIGHT / 2) + CENTER_Y).astype(np.int32)
    points = list(zip(range(CHUNK), ys))   # [(0,y0),(1,y1),...]
    screen.fill(BG_COLOR)
    if len(points) > 1:
        pygame.draw.lines(screen, WAVE_COLOR, False, points, LINE_THICKNESS)
    
    # dD box overlay + bar
        # make a semi-transparent box
    box_w, box_h = 240, 50
    box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    box.fill((0, 0, 0, 140))   # translucent background

        # text
    label = f"Level: {ema_db:.1f} dBFS"
    text_surface = font.render(label, True, TEXT_COLOR)
    box.blit(text_surface, (10, 8))

        # bar
    t = max(0.0, min(1.0, (ema_db + 60.0) / 60.0))  # map [-60..0] -> [0..1]
    bar_w, bar_h = box_w - 20, 12
    x0, y0 = 10, box_h - bar_h - 8
    pygame.draw.rect(box, (80, 80, 80), (x0, y0, bar_w, bar_h), border_radius=6)      # track
    pygame.draw.rect(box, (0, 200, 120), (x0, y0, int(bar_w * t), bar_h), border_radius=6)  # fill

        # blit to screen
    screen.blit(box, (8, 8))

    # ---------------- Draw slider ----------------
    # Track
    pygame.draw.rect(screen, (60, 60, 60), (SLIDER_X, SLIDER_Y - SLIDER_H//2, SLIDER_W, SLIDER_H), border_radius=6)
    # Fill up to handle (just visual)
    handle_x = alpha_to_x(alpha)
    pygame.draw.rect(screen, (0, 160, 220), (SLIDER_X, SLIDER_Y - SLIDER_H//2, max(0, handle_x - SLIDER_X), SLIDER_H), border_radius=6)
    # Handle
    pygame.draw.rect(screen, (220, 220, 220), (handle_x - HANDLE_W//2, SLIDER_Y - 12, HANDLE_W, 24), border_radius=6)
    # Label for slider
    alpha_text = font.render(f"Smoothing (alpha): {alpha:.2f}", True, TEXT_COLOR)
    screen.blit(alpha_text, (SLIDER_X, SLIDER_Y - 30))
    # ------------------------------------------------

    # Flip and cap FPS
    pygame.display.flip()
    clock.tick(FPS)
    # flip() swaps buffers so the new frame shows.
    # tick(60) tries to run ~60 frames/sec (stable animation, less CPU).

# Cleanup
stream.stop_stream(); stream.close(); pa.terminate()
pygame.quit(); sys.exit(0)

# “We open the mic with PyAudio, read 800 samples per frame, convert bytes to a NumPy array, normalize to −1..1, compute RMS→dBFS for a smoothed loudness meter, map samples to screen coordinates, draw a polyline waveform with Pygame, render a readable dB overlay, and repeat at 60 fps.”
