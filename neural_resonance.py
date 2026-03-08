import pygame
import random
import sys
import socket
import time
import math
import threading

# Shared BCI connection status — updated by the background thread, read by HUD
bci_status = "INITIALIZING"   # possible values: INITIALIZING, BUFFERING, LIVE, DEMO, ERROR

# ─── BCI Processing (runs in background thread) ───────────────────────────────
def start_bci_processing():
    """
    Launches processing.py logic in a daemon thread.
    Reads from LSL, computes team sync score, sends to game via UDP.
    Daemon=True means it auto-kills when the game window closes.
    """
    def run():
        global bci_status
        try:
            from pylsl import StreamInlet, resolve_streams
            from scipy.signal import butter, filtfilt, welch
            import numpy as np

            UDP_IP   = "127.0.0.1"
            UDP_PORT = 5005
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            fs             = 250
            WINDOW_SAMPLES = 1250
            UPDATE_EVERY   = 25

            def bandpass(data, low, high, fs, order=4):
                nyq = 0.5 * fs
                b, a = butter(order, [low/nyq, high/nyq], btype='band')
                return filtfilt(b, a, data)

            def band_power(signal, low, high):
                freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
                return np.mean(psd[(freqs >= low) & (freqs <= high)])

            def band_sync(sigA, sigB, low, high):
                A = bandpass(sigA, low, high, fs)
                B = bandpass(sigB, low, high, fs)
                return np.corrcoef(A, B)[0, 1]

            print("[BCI] Looking for EEG streams...")
            streams = resolve_streams()
            if len(streams) == 0:
                print("[BCI] No LSL streams found — running in demo mode.")
                bci_status = "DEMO"
                return

            inlet = StreamInlet(streams[0])
            print(f"[BCI] Connected to stream '{streams[0].name()}'. Collecting data...")
            bci_status = "BUFFERING"

            buffer = []
            samples_since_update = 0

            while True:
                sample, _ = inlet.pull_sample()
                buffer.append(sample)
                samples_since_update += 1

                if len(buffer) > WINDOW_SAMPLES:
                    buffer.pop(0)

                # Still filling the initial window — show buffer progress
                if len(buffer) < WINDOW_SAMPLES:
                    pct = int(100 * len(buffer) / WINDOW_SAMPLES)
                    bci_status = f"BUFFERING {pct}%"
                    continue

                if samples_since_update < UPDATE_EVERY:
                    continue

                samples_since_update = 0
                bci_status = "LIVE"

                data     = np.array(buffer)
                eeg      = data.T
                filtered = np.zeros_like(eeg)
                for ch in range(8):
                    filtered[ch] = bandpass(eeg[ch], 1, 40, fs)

                personA = filtered[0:4]
                personB = filtered[4:8]

                alpha_sync = band_sync(personA[0], personB[0], 8,  12)
                beta_sync  = band_sync(personA[0], personB[0], 13, 30)
                theta_sync = band_sync(personA[0], personB[0], 4,   7)

                team_score = (
                    0.4 * alpha_sync +
                    0.4 * beta_sync  +
                    0.2 * theta_sync
                )

                team_score_clamped = float(np.clip(team_score, 0.0, 1.0))
                out_sock.sendto(str(team_score_clamped).encode(), (UDP_IP, UDP_PORT))

                bar = '█' * int(team_score_clamped * 30)
                print(f"[BCI] α={alpha_sync:+.3f}  β={beta_sync:+.3f}  "
                      f"θ={theta_sync:+.3f}  SCORE={team_score_clamped:.3f}  [{bar:<30}]")

        except Exception as e:
            print(f"[BCI] Error in processing thread: {e}")
            print("[BCI] Falling back to demo mode.")
            bci_status = "ERROR"

    t = threading.Thread(target=run, daemon=True)
    t.start()

# --- Constants & Colors ---
WIDTH, HEIGHT = 1200, 750
FPS = 60
TIME_LIMIT = 60

# Biopunk Neural Palette
BG_DARK       = (4, 6, 14)
BG_MID        = (8, 12, 28)
GRID_COLOR    = (15, 22, 45)
GRID_BRIGHT   = (20, 35, 70)

CYAN          = (0, 220, 255)
CYAN_DIM      = (0, 100, 140)
CYAN_GLOW     = (0, 255, 255)
PINK          = (255, 30, 140)
PINK_DIM      = (140, 15, 75)
PINK_GLOW     = (255, 80, 180)

SYNCED_COLOR  = (80, 255, 160)
SYNCED_DIM    = (30, 100, 65)
WHITE         = (220, 235, 255)
WHITE_DIM     = (100, 120, 160)
AMBER         = (255, 180, 0)
DANGER        = (255, 60, 60)

HUD_BG        = (8, 14, 30, 200)
PANEL_BORDER  = (30, 50, 100)

# --- Networking Setup ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEURAL RESONANCE PROTOCOL  ·  OpenBCI")
clock = pygame.time.Clock()

# Fonts
font_mono_sm  = pygame.font.SysFont("Courier New", 13, bold=True)
font_mono_md  = pygame.font.SysFont("Courier New", 18, bold=True)
font_mono_lg  = pygame.font.SysFont("Courier New", 26, bold=True)
font_mono_xl  = pygame.font.SysFont("Courier New", 48, bold=True)
font_mono_xxl = pygame.font.SysFont("Courier New", 64, bold=True)


# ─── Utility: Glow surface ────────────────────────────────────────────────────
def make_glow(surf, color, radius=12):
    w, h = surf.get_size()
    glow = pygame.Surface((w + radius*2, h + radius*2), pygame.SRCALPHA)
    alpha_per_step = 60 // max(radius // 3, 1)
    for i in range(radius, 0, -3):
        alpha = max(0, 255 - (radius - i) * alpha_per_step)
        tmp = pygame.transform.scale(surf, (w + i*2, h + i*2))
        tmp.set_alpha(int(alpha * 0.3))
        glow.blit(tmp, (radius - i, radius - i))
    glow.blit(surf, (radius, radius))
    return glow


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def alpha_surface(color, w, h, alpha):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color, alpha))
    return s


# ─── Particle System ──────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, vel=None, life=None, size=None):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.vel = vel or [random.uniform(-2, 2), random.uniform(-2, 2)]
        self.life = life or random.randint(20, 50)
        self.max_life = self.life
        self.size = size or random.uniform(1.5, 4.0)

    def update(self):
        self.x += self.vel[0]
        self.y += self.vel[1]
        self.vel[0] *= 0.94
        self.vel[1] *= 0.94
        self.life -= 1

    def draw(self, surf):
        t = self.life / self.max_life
        alpha = int(255 * t)
        r = max(1, int(self.size * t))
        s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (r+1, r+1), r)
        surf.blit(s, (int(self.x) - r - 1, int(self.y) - r - 1))


particles = []

def spawn_capture_burst(x, y, color, count=40):
    for _ in range(count):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(1.5, 6.0)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        particles.append(Particle(x, y, color, vel=vel, life=random.randint(30, 70), size=random.uniform(2, 5)))


def spawn_trail(x, y, color):
    if random.random() < 0.35:
        particles.append(Particle(x, y, color,
            vel=[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
            life=random.randint(8, 20), size=random.uniform(1, 2.5)))


# ─── Scanline Effect ──────────────────────────────────────────────────────────
scanline_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
for y in range(0, HEIGHT, 3):
    pygame.draw.line(scanline_surf, (0, 0, 0, 35), (0, y), (WIDTH, y))


# ─── Grid ─────────────────────────────────────────────────────────────────────
def draw_grid(sync):
    bright_t = min(sync * 1.5, 1.0)
    c = lerp_color(GRID_COLOR, GRID_BRIGHT, bright_t)
    for x in range(0, WIDTH, 60):
        pygame.draw.line(screen, c, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 60):
        pygame.draw.line(screen, c, (0, y), (WIDTH, y))
    # Accent grid lines
    accent = lerp_color(GRID_COLOR, (0, 80, 120), bright_t)
    for x in range(0, WIDTH, 300):
        pygame.draw.line(screen, accent, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 300):
        pygame.draw.line(screen, accent, (0, y), (WIDTH, y))


# ─── Neural Sync Waveform ─────────────────────────────────────────────────────
wave_history = []  # stores recent sync values for waveform

def draw_waveform(x, y, w, h, history, color):
    if len(history) < 2:
        return
    # Background panel
    panel = alpha_surface(BG_MID, w, h, 180)
    screen.blit(panel, (x, y))
    pygame.draw.rect(screen, PANEL_BORDER, (x, y, w, h), 1)

    # Draw the wave
    step = w / max(len(history) - 1, 1)
    pts = []
    for i, v in enumerate(history):
        px = x + int(i * step)
        py = y + h - int(v * (h - 6)) - 3
        pts.append((px, py))

    if len(pts) >= 2:
        # Glow passes
        for thickness, alpha_factor in [(5, 0.2), (3, 0.5), (1, 1.0)]:
            s_tmp = pygame.Surface((w, h), pygame.SRCALPHA)
            adjusted = [(p[0] - x, p[1] - y) for p in pts]
            if len(adjusted) >= 2:
                pygame.draw.lines(s_tmp, (*color, int(255 * alpha_factor)), False, adjusted, thickness)
            screen.blit(s_tmp, (x, y))

    # Fill area under curve
    fill_pts = [(pts[0][0], y + h)] + pts + [(pts[-1][0], y + h)]
    fill_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    adjusted_fill = [(p[0] - x, p[1] - y) for p in fill_pts]
    if len(adjusted_fill) >= 3:
        pygame.draw.polygon(fill_surf, (*color, 30), adjusted_fill)
    screen.blit(fill_surf, (x, y))


# ─── Sync Ring (central indicator) ───────────────────────────────────────────
def draw_sync_ring(cx, cy, sync, tick):
    base_r = 60
    r = int(base_r + sync * 20)
    # Outer pulse rings
    for i in range(3):
        phase = (tick * 0.04 + i * 0.4) % 1.0
        pr = int(r + phase * 50)
        alpha = int(120 * (1 - phase))
        ring_s = pygame.Surface((pr*2+4, pr*2+4), pygame.SRCALPHA)
        color = lerp_color(CYAN, SYNCED_COLOR, sync)
        pygame.draw.circle(ring_s, (*color, alpha), (pr+2, pr+2), pr, 1)
        screen.blit(ring_s, (cx - pr - 2, cy - pr - 2))

    # Core ring
    color = lerp_color(CYAN, SYNCED_COLOR, sync)
    for thickness, a in [(10, 40), (6, 80), (3, 160), (1, 255)]:
        ring_s = pygame.Surface((r*2+20, r*2+20), pygame.SRCALPHA)
        pygame.draw.circle(ring_s, (*color, a), (r+10, r+10), r, thickness)
        screen.blit(ring_s, (cx - r - 10, cy - r - 10))

    # Arc fill based on sync
    arc_rect = pygame.Rect(cx - r, cy - r, r*2, r*2)
    if sync > 0.01:
        arc_surf = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.arc(arc_surf, (*color, 100), (1, 1, r*2, r*2),
                        math.pi/2, math.pi/2 + math.tau * sync, 4)
        screen.blit(arc_surf, (cx - r - 1, cy - r - 1))

    # Inner percentage
    pct_text = font_mono_lg.render(f"{int(sync*100)}%", True, color)
    screen.blit(pct_text, (cx - pct_text.get_width()//2, cy - pct_text.get_height()//2))
    label = font_mono_sm.render("NEURAL SYNC", True, WHITE_DIM)
    screen.blit(label, (cx - label.get_width()//2, cy + 28))


# ─── Target ───────────────────────────────────────────────────────────────────
def draw_target(tx, ty, sync, tick):
    size = 28
    pulse = 1.0 + 0.15 * math.sin(tick * 0.1)
    ps = int(size * pulse)
    color = lerp_color(WHITE, SYNCED_COLOR, sync)

    # Outer glow
    for r_off, alpha in [(20, 15), (12, 35), (6, 70)]:
        gs = pygame.Surface(((ps + r_off)*2, (ps + r_off)*2), pygame.SRCALPHA)
        pygame.draw.rect(gs, (*color, alpha),
                         (0, 0, (ps+r_off)*2, (ps+r_off)*2), 0)
        screen.blit(gs, (tx - ps - r_off, ty - ps - r_off))

    # Crosshair lines (short, diagnostic style)
    gap = ps + 6
    pygame.draw.line(screen, color, (tx - gap - 15, ty), (tx - gap, ty), 2)
    pygame.draw.line(screen, color, (tx + gap, ty), (tx + gap + 15, ty), 2)
    pygame.draw.line(screen, color, (tx, ty - gap - 15), (tx, ty - gap), 2)
    pygame.draw.line(screen, color, (tx, ty + gap), (tx, ty + gap + 15), 2)

    # Box
    pygame.draw.rect(screen, color, (tx - ps, ty - ps, ps*2, ps*2), 2)
    # Inner dot
    pygame.draw.circle(screen, AMBER, (tx, ty), 5)
    pygame.draw.circle(screen, AMBER, (tx, ty), 3)

    # Rotating corner brackets
    bsize = ps + 4
    angle_offset = tick * 0.02
    for i, (dx, dy) in enumerate([(1,1),(-1,1),(1,-1),(-1,-1)]):
        corner_x = tx + dx * bsize
        corner_y = ty + dy * bsize
        for bx, by in [(dx * 10, 0), (0, dy * 10)]:
            pygame.draw.line(screen, AMBER,
                             (corner_x, corner_y),
                             (corner_x + bx, corner_y + by), 2)


# ─── Player Crosshair ─────────────────────────────────────────────────────────
def draw_player(px, py, sync, tick):
    c1 = lerp_color(CYAN, SYNCED_COLOR, sync)
    c2 = lerp_color(PINK, SYNCED_COLOR, sync)
    alpha = int(160 + sync * 60)

    # X-axis line (cyan)
    x_surf = pygame.Surface((WIDTH, 2), pygame.SRCALPHA)
    x_surf.fill((*c1, alpha))
    screen.blit(x_surf, (0, py - 1))

    # Y-axis line (pink)
    y_surf = pygame.Surface((2, HEIGHT), pygame.SRCALPHA)
    y_surf.fill((*c2, alpha))
    screen.blit(y_surf, (px - 1, 0))

    # Crosshair glow lines (thicker, transparent)
    for thickness, a in [(8, 20), (4, 50)]:
        gx = pygame.Surface((WIDTH, thickness), pygame.SRCALPHA)
        gx.fill((*c1, a))
        screen.blit(gx, (0, py - thickness//2))
        gy = pygame.Surface((thickness, HEIGHT), pygame.SRCALPHA)
        gy.fill((*c2, a))
        screen.blit(gy, (px - thickness//2, 0))

    # Center box
    box_size = 22
    pygame.draw.rect(screen, c1, (px - box_size, py - box_size, box_size*2, box_size*2), 2)
    # Corner ticks
    tick_len = 8
    for dx, dy in [(1,1),(-1,1),(1,-1),(-1,-1)]:
        cx = px + dx * box_size
        cy = py + dy * box_size
        pygame.draw.line(screen, WHITE, (cx, cy), (cx + dx * tick_len, cy), 1)
        pygame.draw.line(screen, WHITE, (cx, cy), (cx, cy + dy * tick_len), 1)

    # Center dot
    pygame.draw.circle(screen, WHITE, (px, py), 4)
    pygame.draw.circle(screen, BG_DARK, (px, py), 2)


# ─── HUD Panels ───────────────────────────────────────────────────────────────
def draw_top_hud(remaining, score, sync, tick):
    # Top bar background
    bar = alpha_surface(BG_MID, WIDTH, 58, 210)
    screen.blit(bar, (0, 0))
    pygame.draw.line(screen, PANEL_BORDER, (0, 58), (WIDTH, 58), 1)
    # Accent bar
    accent_w = int(WIDTH * (1 - remaining/TIME_LIMIT))
    if accent_w > 0:
        time_color = lerp_color(SYNCED_COLOR, DANGER, max(0, 1 - remaining/15))
        pygame.draw.rect(screen, (*time_color, 120), (0, 56, accent_w, 2))

    # Left: Title
    title = font_mono_md.render("NEURAL RESONANCE PROTOCOL", True, CYAN)
    screen.blit(title, (20, 10))
    sub = font_mono_sm.render("OpenBCI  ·  EEG INTER-BRAIN SYNC", True, CYAN_DIM)
    screen.blit(sub, (20, 34))

    # BCI status indicator
    status_colors = {
        "LIVE":         (SYNCED_COLOR, (10, 40, 25)),
        "DEMO":         (AMBER,        (40, 30, 5)),
        "ERROR":        (DANGER,       (40, 8, 8)),
        "INITIALIZING": (WHITE_DIM,    (20, 25, 45)),
    }
    status_key = ("LIVE" if bci_status == "LIVE"
                  else "INITIALIZING" if bci_status.startswith("BUFFERING")
                  else bci_status if bci_status in status_colors
                  else "ERROR")
    s_fg, _ = status_colors.get(status_key, (WHITE_DIM, (20, 25, 45)))
    dot_x = 282
    # Blink dot when LIVE, solid otherwise
    show_dot = not (bci_status == "LIVE" and (tick // 20) % 2 == 0)
    if show_dot:
        pygame.draw.circle(screen, s_fg, (dot_x, 41), 5)
    else:
        pygame.draw.circle(screen, s_fg, (dot_x, 41), 5, 1)
    status_surf = font_mono_sm.render(bci_status, True, s_fg)
    screen.blit(status_surf, (dot_x + 10, 34))

    # Center: Timer
    t_color = lerp_color(WHITE, DANGER, max(0, 1 - remaining/15))
    time_str = f"{int(remaining):02d}s"
    tsurf = font_mono_xl.render(time_str, True, t_color)
    screen.blit(tsurf, (WIDTH//2 - tsurf.get_width()//2, 4))

    # Right: Score
    score_label = font_mono_sm.render("CAPTURES", True, WHITE_DIM)
    score_val   = font_mono_xl.render(str(score), True, SYNCED_COLOR)
    screen.blit(score_label, (WIDTH - 160, 8))
    screen.blit(score_val, (WIDTH - 160, 24))

    # Blinking REC dot
    if (tick // 30) % 2 == 0:
        pygame.draw.circle(screen, DANGER, (WIDTH - 25, 15), 6)
        rec = font_mono_sm.render("REC", True, DANGER)
        screen.blit(rec, (WIDTH - 55, 8))


def draw_bottom_hud(sync, sync_history, tick):
    panel_h = 110
    y0 = HEIGHT - panel_h
    bar = alpha_surface(BG_MID, WIDTH, panel_h, 210)
    screen.blit(bar, (0, y0))
    pygame.draw.line(screen, PANEL_BORDER, (0, y0), (WIDTH, y0), 1)

    # Left labels: P1 / P2
    p1_color = lerp_color(CYAN, SYNCED_COLOR, sync)
    p2_color = lerp_color(PINK, SYNCED_COLOR, sync)
    p1 = font_mono_md.render("▶  P1  [A / D]", True, p1_color)
    p2 = font_mono_md.render("▶  P2  [↑ / ↓]", True, p2_color)
    screen.blit(p1, (20, y0 + 15))
    screen.blit(p2, (20, y0 + 50))

    # Sync bar
    bar_x, bar_y, bar_w, bar_h = 250, y0 + 30, 300, 22
    pygame.draw.rect(screen, (20, 30, 60), (bar_x, bar_y, bar_w, bar_h), 0)
    pygame.draw.rect(screen, PANEL_BORDER, (bar_x, bar_y, bar_w, bar_h), 1)
    fill_w = int(bar_w * sync)
    if fill_w > 0:
        bar_color = lerp_color(CYAN, SYNCED_COLOR, sync)
        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
        # Shine
        shine = alpha_surface(WHITE, fill_w, bar_h//2, 40)
        screen.blit(shine, (bar_x, bar_y))
    label = font_mono_sm.render(f"SYNC LEVEL  {int(sync*100):3d}%", True, WHITE_DIM)
    screen.blit(label, (bar_x, bar_y - 16))

    # State label
    if sync > 0.75:
        state, sc = "● HYPER-SYNC", SYNCED_COLOR
    elif sync > 0.5:
        state, sc = "◆ IN-SYNC", CYAN
    elif sync > 0.25:
        state, sc = "◇ PARTIAL SYNC", AMBER
    else:
        state, sc = "○ ASYNC", WHITE_DIM
    state_surf = font_mono_md.render(state, True, sc)
    screen.blit(state_surf, (bar_x, bar_y + 30))

    # Waveform (right side)
    wf_x, wf_y, wf_w, wf_h = 620, y0 + 10, 560, panel_h - 20
    recent = sync_history[-120:] if len(sync_history) > 120 else sync_history
    draw_waveform(wf_x, wf_y, wf_w, wf_h, recent, lerp_color(CYAN, SYNCED_COLOR, sync))
    wf_label = font_mono_sm.render("EEG COHERENCE  (ROLLING 2s)", True, WHITE_DIM)
    screen.blit(wf_label, (wf_x + 4, wf_y + 4))


def draw_side_stats(sync, score, tick):
    # Right side mini panel
    px, py, pw, ph = WIDTH - 220, 80, 200, 200
    panel = alpha_surface(BG_MID, pw, ph, 180)
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, PANEL_BORDER, (px, py, pw, ph), 1)

    items = [
        ("FREQ BAND",   "ALPHA"),
        ("ELECTRODE",   "Fp1/Fp2"),
        ("COHERENCE",   f"{int(sync*100)}%"),
        ("CAPTURES",    str(score)),
        ("TICK",        str(tick % 10000)),
    ]
    for i, (label, val) in enumerate(items):
        l_surf = font_mono_sm.render(label, True, WHITE_DIM)
        v_color = lerp_color(WHITE_DIM, SYNCED_COLOR, sync) if label == "COHERENCE" else WHITE
        v_surf  = font_mono_sm.render(val, True, v_color)
        row_y = py + 14 + i * 34
        screen.blit(l_surf, (px + 10, row_y))
        screen.blit(v_surf,  (px + pw - v_surf.get_width() - 10, row_y))
        if i < len(items) - 1:
            pygame.draw.line(screen, PANEL_BORDER, (px+8, row_y + 20), (px + pw - 8, row_y + 20), 1)


# ─── End Screen ───────────────────────────────────────────────────────────────
def draw_end_screen(screen, score, avg_sync, max_sync, tick):
    screen.fill(BG_DARK)
    draw_grid(avg_sync)
    screen.blit(scanline_surf, (0, 0))

    # Central panel
    pw, ph = 700, 400
    px, py = WIDTH//2 - pw//2, HEIGHT//2 - ph//2
    panel = alpha_surface(BG_MID, pw, ph, 230)
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, PANEL_BORDER, (px, py, pw, ph), 2)
    # Accent top bar
    pygame.draw.rect(screen, lerp_color(CYAN, SYNCED_COLOR, avg_sync), (px, py, pw, 3))

    title = font_mono_xl.render("SESSION COMPLETE", True, WHITE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, py + 30))

    sub = font_mono_md.render("INTER-BRAIN SYNCHRONIZATION REPORT", True, CYAN_DIM)
    screen.blit(sub, (WIDTH//2 - sub.get_width()//2, py + 90))
    pygame.draw.line(screen, PANEL_BORDER, (px + 40, py + 120), (px + pw - 40, py + 120), 1)

    # Stats grid
    stats = [
        ("AVG NEURAL COHERENCE", f"{int(avg_sync*100)}%", lerp_color(CYAN, SYNCED_COLOR, avg_sync)),
        ("PEAK SYNC",            f"{int(max_sync*100)}%", SYNCED_COLOR),
        ("TOTAL CAPTURES",       str(score),              AMBER),
        ("SESSION DURATION",     f"{TIME_LIMIT}s",        WHITE),
    ]
    col_w = (pw - 80) // 2
    for i, (label, val, color) in enumerate(stats):
        col = i % 2
        row = i // 2
        sx = px + 40 + col * (col_w + 40)
        sy = py + 145 + row * 90
        l_surf = font_mono_sm.render(label, True, WHITE_DIM)
        v_surf = font_mono_lg.render(val, True, color)
        screen.blit(l_surf, (sx, sy))
        screen.blit(v_surf, (sx, sy + 18))

    pygame.draw.line(screen, PANEL_BORDER, (px + 40, py + 330), (px + pw - 40, py + 330), 1)
    footer = font_mono_sm.render("PRESS  ESC  TO  EXIT", True, WHITE_DIM)
    screen.blit(footer, (WIDTH//2 - footer.get_width()//2, py + 345))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    player_pos = [WIDTH // 2, HEIGHT // 2]
    target_pos = [random.randint(120, WIDTH-120), random.randint(120, HEIGHT-160)]
    target_speed = 3.0
    target_dir = [random.choice([-1, 1]), random.choice([-1, 1])]

    score = 0
    start_time = time.time()
    neural_sync = 0.0
    sync_history = []
    tick = 0
    max_sync = 0.0
    game_over = False

    running = True
    while running:
        tick += 1
        elapsed_time = time.time() - start_time
        remaining_time = max(0, TIME_LIMIT - elapsed_time)

        if remaining_time <= 0 and not game_over:
            game_over = True

        # ── Receive Brain Data ────────────────────────────────────────────────
        try:
            data, addr = sock.recvfrom(1024)
            neural_sync = float(data.decode())
            neural_sync = max(0.0, min(1.0, neural_sync))
            sync_history.append(neural_sync)
            if neural_sync > max_sync:
                max_sync = neural_sync
        except:
            # Demo: slowly drift sync value when no data
            neural_sync = 0.5 + 0.4 * math.sin(tick * 0.02)
            neural_sync = max(0.0, min(1.0, neural_sync))
            sync_history.append(neural_sync)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # ── Game Over Screen ──────────────────────────────────────────────────
        if game_over:
            avg_sync = sum(sync_history)/len(sync_history) if sync_history else 0
            draw_end_screen(screen, score, avg_sync, max_sync, tick)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ── Player Movement ───────────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        speed = 5 + (neural_sync * 9)
        # P1 (cyan, X-axis): A / D keys — left side of keyboard
        if keys[pygame.K_a]: player_pos[0] -= speed
        if keys[pygame.K_d]: player_pos[0] += speed
        # P2 (pink, Y-axis): UP / DOWN arrow keys — right side of keyboard
        if keys[pygame.K_UP]:   player_pos[1] -= speed
        if keys[pygame.K_DOWN]: player_pos[1] += speed
        player_pos[0] = max(0, min(WIDTH, player_pos[0]))
        player_pos[1] = max(60, min(HEIGHT - 120, player_pos[1]))

        # ── Target Movement ───────────────────────────────────────────────────
        current_target_speed = target_speed * (1.3 - neural_sync)
        target_pos[0] += current_target_speed * target_dir[0]
        target_pos[1] += current_target_speed * target_dir[1]
        if target_pos[0] <= 60 or target_pos[0] >= WIDTH - 60:   target_dir[0] *= -1
        if target_pos[1] <= 70 or target_pos[1] >= HEIGHT - 130: target_dir[1] *= -1

        # ── Capture Logic ─────────────────────────────────────────────────────
        dist = math.hypot(player_pos[0] - target_pos[0], player_pos[1] - target_pos[1])
        if dist < 45:
            score += 1
            capture_color = lerp_color(CYAN, SYNCED_COLOR, neural_sync)
            spawn_capture_burst(target_pos[0], target_pos[1], capture_color)
            target_pos = [random.randint(100, WIDTH-120), random.randint(80, HEIGHT-150)]

        # ── Trails ────────────────────────────────────────────────────────────
        spawn_trail(player_pos[0], player_pos[1], lerp_color(CYAN, SYNCED_COLOR, neural_sync))
        spawn_trail(target_pos[0], target_pos[1], lerp_color(PINK_DIM, SYNCED_COLOR, neural_sync * 0.5))

        # ── Update Particles ──────────────────────────────────────────────────
        for p in particles[:]:
            p.update()
            if p.life <= 0:
                particles.remove(p)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG_DARK)
        draw_grid(neural_sync)

        # Background vignette
        vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for r in range(500, 0, -20):
            alpha = max(0, int(60 * (1 - r / 500)))
            pygame.draw.circle(vignette, (0, 0, 0, alpha), (WIDTH//2, HEIGHT//2), r)
        # Apply as darkening at edges - reverse vignette
        edge_vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        edge_vignette.fill((0, 0, 0, 0))
        pygame.draw.rect(edge_vignette, (0, 0, 20, 60), (0, 0, WIDTH, HEIGHT))
        pygame.draw.circle(edge_vignette, (0, 0, 0, 0), (WIDTH//2, HEIGHT//2), 500)
        screen.blit(edge_vignette, (0, 0))

        # Particles
        for p in particles:
            p.draw(screen)

        draw_target(int(target_pos[0]), int(target_pos[1]), neural_sync, tick)
        draw_player(int(player_pos[0]), int(player_pos[1]), neural_sync, tick)

        # HUD
        draw_top_hud(remaining_time, score, neural_sync, tick)
        draw_bottom_hud(neural_sync, sync_history, tick)
        draw_side_stats(neural_sync, score, tick)

        # Scanlines overlay
        screen.blit(scanline_surf, (0, 0))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    start_bci_processing()
    main()