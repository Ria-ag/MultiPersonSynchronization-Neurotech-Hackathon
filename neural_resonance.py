import pygame
import random
import sys
import socket
import time
import math
import threading

# Shared BCI connection status — updated by the background thread, read by HUD
bci_status = "INITIALIZING"   # possible values: INITIALIZING, BUFFERING, LIVE, DEMO, ERROR

# Raw EEG samples accumulated across the full session for end-screen coherence calculation
# Each entry is (p1_sample, p2_sample) — one scalar pair per Cyton sample
session_eeg_buffer = []
session_eeg_lock   = threading.Lock()

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

                # Accumulate raw signal pairs for full-session coherence at end screen
                with session_eeg_lock:
                    session_eeg_buffer.append((sample[0], sample[4]))  # P1 ch0, P2 ch0

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
                out_sock.sendto(
                    f"{team_score_clamped:.4f},{alpha_sync:.4f},{beta_sync:.4f},{theta_sync:.4f}".encode(),
                    (UDP_IP, UDP_PORT))

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
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

# ─── NOTE: sock is created inside main() so a busy port on restart
#     doesn't crash the script before the pygame window opens. ─────────────────

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


def draw_side_stats(sync, score, tick, band_vals):
    px, py, pw, ph = WIDTH - 220, 80, 200, 200
    panel = alpha_surface(BG_MID, pw, ph, 180)
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, PANEL_BORDER, (px, py, pw, ph), 1)

    items = [
        ("COMPOSITE",  f"{int(sync*100)}%",                       sync),
        ("α ALPHA",    f"{int(band_vals['alpha']*100):+d}%",      max(0, band_vals['alpha'])),
        ("β BETA",     f"{int(band_vals['beta']*100):+d}%",       max(0, band_vals['beta'])),
        ("θ THETA",    f"{int(band_vals['theta']*100):+d}%",      max(0, band_vals['theta'])),
        ("CAPTURES",   str(score),                                 0.5),
    ]
    for i, (label, val, t) in enumerate(items):
        l_surf = font_mono_sm.render(label, True, WHITE_DIM)
        v_color = lerp_color(WHITE_DIM, SYNCED_COLOR, min(t, 1.0))
        v_surf  = font_mono_sm.render(val, True, v_color)
        row_y = py + 14 + i * 34
        screen.blit(l_surf, (px + 10, row_y))
        screen.blit(v_surf,  (px + pw - v_surf.get_width() - 10, row_y))
        if i < len(items) - 1:
            pygame.draw.line(screen, PANEL_BORDER, (px+8, row_y + 20), (px + pw - 8, row_y + 20), 1)


# ─── Full-session coherence (computed once at game over) ──────────────────────
def compute_session_coherence():
    """
    Magnitude-squared coherence averaged across the alpha band (8-12 Hz),
    computed over the entire session's raw EEG data.

    This is a proper frequency-domain measure — different from the real-time
    Pearson correlation used for the live sync score.

    Returns a float in [0, 1], or None if not enough data.
    """
    with session_eeg_lock:
        if len(session_eeg_buffer) < 500:   # need at least 2 seconds
            return None
        pairs = list(session_eeg_buffer)    # snapshot

    try:
        from scipy.signal import coherence, butter, filtfilt
        import numpy as np

        fs = 250
        sig_p1 = np.array([p[0] for p in pairs], dtype=np.float64)
        sig_p2 = np.array([p[1] for p in pairs], dtype=np.float64)

        # Broadband filter first (1-40 Hz) — same as processing pipeline
        nyq = fs / 2.0
        b, a = butter(4, [1/nyq, 40/nyq], btype='band')
        sig_p1 = filtfilt(b, a, sig_p1)
        sig_p2 = filtfilt(b, a, sig_p2)

        # Magnitude-squared coherence across full signal
        nperseg = min(512, len(sig_p1) // 4)
        freqs, Cxy = coherence(sig_p1, sig_p2, fs=fs, nperseg=nperseg)

        # Average across alpha band (8-12 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_coh  = float(np.mean(Cxy[alpha_mask])) if np.any(alpha_mask) else 0.0

        # Also compute beta and theta for breakdown display
        beta_mask  = (freqs >= 13) & (freqs <= 30)
        theta_mask = (freqs >= 4)  & (freqs <= 7)
        beta_coh   = float(np.mean(Cxy[beta_mask]))  if np.any(beta_mask)  else 0.0
        theta_coh  = float(np.mean(Cxy[theta_mask])) if np.any(theta_mask) else 0.0

        # Weighted composite — same bands as live score for comparability
        composite = 0.4 * alpha_coh + 0.4 * beta_coh + 0.2 * theta_coh

        return {
            "composite": float(np.clip(composite, 0, 1)),
            "alpha":     float(np.clip(alpha_coh,  0, 1)),
            "beta":      float(np.clip(beta_coh,   0, 1)),
            "theta":     float(np.clip(theta_coh,  0, 1)),
            "n_samples": len(pairs),
        }
    except Exception as e:
        print(f"[BCI] Coherence calculation failed: {e}")
        return None


# ─── Capture-sync correlation analysis & HTML export ─────────────────────────
def analyse_and_export(capture_log, sync_timeline, coherence_data, avg_sync, score, difficulty="MEDIUM"):
    import csv, os, json

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base      = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(base, f"session_{difficulty}_{timestamp}.csv")
    html_path = os.path.join(base, f"session_{difficulty}_{timestamp}.html")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    capture_times = {t for t, _ in capture_log}
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "neural_sync", "capture", "difficulty"])
        for t, s in sync_timeline:
            writer.writerow([t, s, 1 if t in capture_times else 0, difficulty])

    # ── Compute findings ──────────────────────────────────────────────────────
    findings = {
        "csv_path":         csv_path,
        "html_path":        html_path,
        "capture_avg_sync": None,
        "noncap_avg_sync":  None,
        "sync_effect":      None,
        "interpretation":   "INSUFFICIENT DATA",
        "n_captures":       score,
        "difficulty":       difficulty,
    }

    if len(capture_log) >= 2 and len(sync_timeline) >= 5:
        capture_syncs    = [s for _, s in capture_log]
        capture_set      = set(round(t) for t, _ in capture_log)
        noncapture_syncs = [s for t, s in sync_timeline if round(t) not in capture_set]
        all_syncs        = [s for _, s in sync_timeline]

        cap_avg  = sum(capture_syncs)    / len(capture_syncs)
        non_avg  = (sum(noncapture_syncs) / len(noncapture_syncs)) if noncapture_syncs else sum(all_syncs)/len(all_syncs)
        effect   = cap_avg - non_avg

        findings["capture_avg_sync"] = round(cap_avg, 3)
        findings["noncap_avg_sync"]  = round(non_avg, 3)
        findings["sync_effect"]      = round(effect, 3)

        if abs(effect) < 0.03:
            findings["interpretation"] = "NO SIGNIFICANT EFFECT"
        elif effect > 0:
            strength = "STRONG" if effect > 0.10 else "MODERATE" if effect > 0.05 else "WEAK"
            findings["interpretation"] = f"{strength} POSITIVE CORRELATION"
        else:
            findings["interpretation"] = "NEGATIVE CORRELATION"

    print(f"[RESEARCH] CSV exported  → {csv_path}")
    print(f"[RESEARCH] Finding: {findings['interpretation']}  Δ={findings['sync_effect']}")
    return findings


# ─── Native in-game chart rendering ──────────────────────────────────────────

def _chart_coords(val, t, chart_x, chart_y, chart_w, chart_h, t_min, t_max, v_min=0.0, v_max=1.0):
    """Map (time, value) to pixel (px, py) inside the chart area."""
    t_range = max(t_max - t_min, 1e-6)
    v_range = max(v_max - v_min, 1e-6)
    px = chart_x + int((t - t_min) / t_range * chart_w)
    py = chart_y + chart_h - int((val - v_min) / v_range * chart_h)
    return px, py


def draw_chart_axes(chart_x, chart_y, chart_w, chart_h, t_min, t_max,
                    x_label="TIME (s)", y_label="SYNC", y_ticks=5):
    """Draw axis lines, tick marks and labels for a chart area."""
    # Background
    bg = alpha_surface(BG_MID, chart_w, chart_h, 200)
    screen.blit(bg, (chart_x, chart_y))
    pygame.draw.rect(screen, PANEL_BORDER, (chart_x, chart_y, chart_w, chart_h), 1)

    # Horizontal grid + y-axis ticks
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        gy = chart_y + chart_h - int(frac * chart_h)
        col = GRID_BRIGHT if i % y_ticks == 0 else GRID_COLOR
        pygame.draw.line(screen, col, (chart_x, gy), (chart_x + chart_w, gy), 1)
        label = font_mono_sm.render(f"{int(frac*100):3d}%", True, WHITE_DIM)
        screen.blit(label, (chart_x - label.get_width() - 4, gy - 7))

    # Vertical grid + x-axis ticks (every 10 s)
    duration = t_max - t_min
    step = 10 if duration <= 60 else 20
    t = math.ceil(t_min / step) * step
    while t <= t_max:
        gx = chart_x + int((t - t_min) / max(duration, 1) * chart_w)
        pygame.draw.line(screen, GRID_COLOR, (gx, chart_y), (gx, chart_y + chart_h), 1)
        lbl = font_mono_sm.render(f"{int(t)}s", True, WHITE_DIM)
        screen.blit(lbl, (gx - lbl.get_width()//2, chart_y + chart_h + 4))
        t += step

    # Axis labels
    xl = font_mono_sm.render(x_label, True, WHITE_DIM)
    screen.blit(xl, (chart_x + chart_w//2 - xl.get_width()//2, chart_y + chart_h + 18))


def draw_timeline_chart(chart_x, chart_y, chart_w, chart_h,
                         sync_timeline, capture_log, findings):
    """
    Full session sync timeline + capture event markers + reference lines.
    Equivalent to the Plotly 'chart' div from the old HTML export.
    """
    if len(sync_timeline) < 2:
        msg = font_mono_md.render("INSUFFICIENT DATA FOR CHART", True, WHITE_DIM)
        screen.blit(msg, (chart_x + chart_w//2 - msg.get_width()//2,
                           chart_y + chart_h//2))
        return

    times = [t for t, _ in sync_timeline]
    syncs = [s for _, s in sync_timeline]
    t_min, t_max = times[0], times[-1]

    draw_chart_axes(chart_x, chart_y, chart_w, chart_h, t_min, t_max)

    def to_px(t, v):
        return _chart_coords(v, t, chart_x, chart_y, chart_w, chart_h, t_min, t_max)

    # ── Raw sync line (dim) ────────────────────────────────────────────────────
    raw_pts = [to_px(t, s) for t, s in sync_timeline]
    if len(raw_pts) >= 2:
        raw_surf = pygame.Surface((chart_w, chart_h), pygame.SRCALPHA)
        adj = [(p[0]-chart_x, p[1]-chart_y) for p in raw_pts]
        pygame.draw.lines(raw_surf, (*CYAN_DIM, 90), False, adj, 1)
        screen.blit(raw_surf, (chart_x, chart_y))

    # ── Smoothed line (5-point rolling avg) ───────────────────────────────────
    def rolling(arr, w=5):
        return [sum(arr[max(0,i-w):i+w+1]) / len(arr[max(0,i-w):i+w+1]) for i in range(len(arr))]
    smooth = rolling(syncs)
    smooth_pts = [to_px(times[i], smooth[i]) for i in range(len(times))]

    # Glow passes for smooth line
    for thickness, alpha in [(5, 30), (3, 70), (1, 220)]:
        s_tmp = pygame.Surface((chart_w, chart_h), pygame.SRCALPHA)
        adj = [(p[0]-chart_x, p[1]-chart_y) for p in smooth_pts]
        if len(adj) >= 2:
            pygame.draw.lines(s_tmp, (*CYAN, alpha), False, adj, thickness)
        screen.blit(s_tmp, (chart_x, chart_y))

    # Fill under smooth line
    fill_pts = [(chart_x, chart_y + chart_h)] + smooth_pts + [(smooth_pts[-1][0], chart_y + chart_h)]
    fill_surf = pygame.Surface((chart_w, chart_h), pygame.SRCALPHA)
    adj_fill = [(p[0]-chart_x, p[1]-chart_y) for p in fill_pts]
    if len(adj_fill) >= 3:
        pygame.draw.polygon(fill_surf, (*CYAN, 25), adj_fill)
    screen.blit(fill_surf, (chart_x, chart_y))

    # ── Reference lines: avg at captures / avg without ────────────────────────
    if findings and findings["capture_avg_sync"] is not None:
        cap_avg = findings["capture_avg_sync"]
        non_avg = findings["noncap_avg_sync"]
        for val, color, label_str in [
            (cap_avg, SYNCED_COLOR, f"AVG AT CAPTURES  {int(cap_avg*100)}%"),
            (non_avg, WHITE_DIM,   f"AVG WITHOUT  {int(non_avg*100)}%"),
        ]:
            ry = chart_y + chart_h - int(val * chart_h)
            ref_surf = pygame.Surface((chart_w, 1), pygame.SRCALPHA)
            # Dashed line via segments
            dash_surf = pygame.Surface((chart_w, chart_h), pygame.SRCALPHA)
            dash_len, gap_len = 12, 6
            x_pos = 0
            while x_pos < chart_w:
                end = min(x_pos + dash_len, chart_w)
                pygame.draw.line(dash_surf, (*color, 160),
                                 (x_pos, ry - chart_y), (end, ry - chart_y), 1)
                x_pos += dash_len + gap_len
            screen.blit(dash_surf, (chart_x, chart_y))
            lbl = font_mono_sm.render(label_str, True, color)
            screen.blit(lbl, (chart_x + chart_w - lbl.get_width() - 6, ry - 14))

    # ── Capture event markers (amber diamonds) ─────────────────────────────────
    for ct, cs in capture_log:
        if t_min <= ct <= t_max:
            mx, my = to_px(ct, cs)
            # Diamond shape
            d = 7
            diamond = [(mx, my-d), (mx+d, my), (mx, my+d), (mx-d, my)]
            pygame.draw.polygon(screen, AMBER, diamond)
            pygame.draw.polygon(screen, WHITE, diamond, 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [
        (CYAN,         "SMOOTHED SYNC"),
        (CYAN_DIM,     "RAW SYNC"),
        (AMBER,        "CAPTURE EVENT"),
    ]
    lx = chart_x + 10
    ly = chart_y + 8
    for color, text in legend:
        pygame.draw.rect(screen, color, (lx, ly + 3, 18, 3))
        lt = font_mono_sm.render(text, True, WHITE_DIM)
        screen.blit(lt, (lx + 24, ly))
        lx += lt.get_width() + 50


def draw_bar_chart(chart_x, chart_y, chart_w, chart_h, findings):
    """
    Two-bar comparison: sync at captures vs sync without captures.
    Equivalent to the Plotly 'bars' div from the old HTML export.
    """
    bg = alpha_surface(BG_MID, chart_w, chart_h, 200)
    screen.blit(bg, (chart_x, chart_y))
    pygame.draw.rect(screen, PANEL_BORDER, (chart_x, chart_y, chart_w, chart_h), 1)

    if not findings or findings["capture_avg_sync"] is None:
        msg = font_mono_sm.render("INSUFFICIENT CAPTURE DATA", True, WHITE_DIM)
        screen.blit(msg, (chart_x + chart_w//2 - msg.get_width()//2,
                           chart_y + chart_h//2))
        return

    cap_avg = findings["capture_avg_sync"]
    non_avg = findings["noncap_avg_sync"]
    effect  = findings["sync_effect"]
    interp  = findings["interpretation"]
    interp_color = (SYNCED_COLOR if "POSITIVE" in interp
                    else DANGER    if "NEGATIVE" in interp
                    else WHITE_DIM)

    # Horizontal grid
    for i in range(6):
        frac = i / 5
        gy = chart_y + chart_h - int(frac * chart_h)
        pygame.draw.line(screen, GRID_COLOR, (chart_x, gy), (chart_x + chart_w, gy), 1)
        lbl = font_mono_sm.render(f"{int(frac*100)}%", True, WHITE_DIM)
        screen.blit(lbl, (chart_x - lbl.get_width() - 4, gy - 7))

    bar_w  = chart_w // 5
    gap    = (chart_w - bar_w * 2) // 3
    bars   = [
        (chart_x + gap,           cap_avg, SYNCED_COLOR, "SYNC AT\nCAPTURES"),
        (chart_x + gap*2 + bar_w, non_avg, WHITE_DIM,    "SYNC\nWITHOUT"),
    ]

    for bx, val, color, label_str in bars:
        bar_h_px = int(val * chart_h)
        by = chart_y + chart_h - bar_h_px

        # Bar fill with gradient effect
        for row in range(bar_h_px):
            t_grad = row / max(bar_h_px, 1)
            row_color = lerp_color(lerp_color(color, BG_MID, 0.5), color, t_grad)
            pygame.draw.line(screen, row_color,
                             (bx, chart_y + chart_h - row),
                             (bx + bar_w, chart_y + chart_h - row))

        # Shine overlay
        shine = alpha_surface(WHITE, bar_w, bar_h_px // 3, 30)
        screen.blit(shine, (bx, by))

        # Border
        pygame.draw.rect(screen, color, (bx, by, bar_w, bar_h_px), 2)

        # Value label above bar
        val_lbl = font_mono_lg.render(f"{int(val*100)}%", True, color)
        screen.blit(val_lbl, (bx + bar_w//2 - val_lbl.get_width()//2, by - 28))

        # Axis labels below chart
        for li, line in enumerate(label_str.split("\n")):
            ll = font_mono_sm.render(line, True, WHITE_DIM)
            screen.blit(ll, (bx + bar_w//2 - ll.get_width()//2,
                              chart_y + chart_h + 6 + li * 14))

    # Δ annotation between the bars
    mid_x = chart_x + chart_w // 2
    eff_str = f"Δ = {effect*100:+.1f}%"
    eff_lbl = font_mono_md.render(eff_str, True, interp_color)
    screen.blit(eff_lbl, (mid_x - eff_lbl.get_width()//2, chart_y + 10))
    int_lbl = font_mono_sm.render(interp, True, interp_color)
    screen.blit(int_lbl, (mid_x - int_lbl.get_width()//2, chart_y + 30))


# ─── End Screen — Tab 0: Summary  /  Tab 1: Charts ───────────────────────────
def draw_end_screen(screen, score, avg_sync, max_sync, tick,
                    coherence_data=None, findings=None,
                    sync_timeline=None, capture_log=None, tab=0):

    sync_timeline = sync_timeline or []
    capture_log   = capture_log   or []

    screen.fill(BG_DARK)
    draw_grid(avg_sync * 0.5)
    screen.blit(scanline_surf, (0, 0))

    # ── Shared top header bar ─────────────────────────────────────────────────
    hdr = alpha_surface(BG_MID, WIDTH, 54, 230)
    screen.blit(hdr, (0, 0))
    pygame.draw.line(screen, PANEL_BORDER, (0, 54), (WIDTH, 54), 1)
    pygame.draw.rect(screen, lerp_color(CYAN, SYNCED_COLOR, avg_sync), (0, 52, WIDTH, 2))

    title_surf = font_mono_lg.render("SESSION COMPLETE  ·  INTER-BRAIN SYNCHRONIZATION REPORT", True, WHITE)
    screen.blit(title_surf, (20, 14))

    # Tab buttons (top-right)
    tab_labels = ["[ 1 ]  SUMMARY", "[ 2 ]  CHARTS"]
    for i, tlabel in enumerate(tab_labels):
        active = (i == tab)
        tc = CYAN if active else WHITE_DIM
        ts = font_mono_sm.render(tlabel, True, tc)
        tx = WIDTH - 320 + i * 155
        screen.blit(ts, (tx, 20))
        if active:
            pygame.draw.line(screen, CYAN, (tx, 36), (tx + ts.get_width(), 36), 2)

    hint = font_mono_sm.render("PRESS  1 / 2  TO SWITCH TABS  ·  ESC TO EXIT", True, (50, 70, 110))
    screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT - 22))

    # ─────────────────────────────────────────────────────────────────────────
    if tab == 0:
        _draw_end_summary(screen, score, avg_sync, max_sync, tick,
                          coherence_data, findings)
    else:
        _draw_end_charts(screen, sync_timeline, capture_log, findings, avg_sync, tick)


def _draw_end_summary(screen, score, avg_sync, max_sync, tick, coherence_data, findings):
    """Tab 0 — the original stats/coherence/findings panel, now full-width."""
    has_findings = findings and findings["sync_effect"] is not None
    pw = 1120
    ph = 560
    px = WIDTH//2 - pw//2
    py = 66

    panel = alpha_surface(BG_MID, pw, ph, 220)
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, PANEL_BORDER, (px, py, pw, ph), 1)

    # ── Top 4 stats in a row ──────────────────────────────────────────────────
    coh_val   = f"{int(coherence_data['composite']*100)}%" if coherence_data else "N/A"
    coh_color = lerp_color(CYAN, SYNCED_COLOR, coherence_data['composite']) if coherence_data else WHITE_DIM

    stats = [
        ("NEURAL COHERENCE", coh_val,                coh_color),
        ("AVG LIVE SYNC",    f"{int(avg_sync*100)}%", lerp_color(CYAN, WHITE, 0.5)),
        ("PEAK SYNC",        f"{int(max_sync*100)}%", SYNCED_COLOR),
        ("TOTAL CAPTURES",   str(score),              AMBER),
    ]
    col_w = pw // 4
    for i, (label, val, color) in enumerate(stats):
        sx = px + i * col_w + col_w//2
        sy = py + 18
        lbl = font_mono_sm.render(label, True, WHITE_DIM)
        val_s = font_mono_xl.render(val, True, color)
        screen.blit(lbl,  (sx - lbl.get_width()//2,  sy))
        screen.blit(val_s,(sx - val_s.get_width()//2, sy + 18))
        if i < 3:
            pygame.draw.line(screen, PANEL_BORDER,
                             (px + (i+1)*col_w, py + 10),
                             (px + (i+1)*col_w, py + 80), 1)

    y_cursor = py + 90
    pygame.draw.line(screen, PANEL_BORDER, (px+20, y_cursor), (px+pw-20, y_cursor), 1)
    y_cursor += 12

    # ── Coherence breakdown ───────────────────────────────────────────────────
    if coherence_data:
        screen.blit(font_mono_sm.render(
            f"COHERENCE BREAKDOWN  ·  {coherence_data['n_samples']} SAMPLES  ·  MAGNITUDE-SQUARED",
            True, WHITE_DIM), (px+20, y_cursor))
        y_cursor += 20

        bands = [
            ("α  ALPHA  8-12Hz", coherence_data['alpha'], CYAN),
            ("β  BETA  13-30Hz", coherence_data['beta'],  PINK),
            ("θ  THETA   4-7Hz", coherence_data['theta'], AMBER),
        ]
        track_x = px + 220
        track_w = pw - 280
        for blabel, bval, bcolor in bands:
            screen.blit(font_mono_sm.render(blabel, True, WHITE_DIM), (px+20, y_cursor))
            pygame.draw.rect(screen, (20,30,60),   (track_x, y_cursor+2, track_w, 14))
            pygame.draw.rect(screen, PANEL_BORDER, (track_x, y_cursor+2, track_w, 14), 1)
            fill = int(track_w * max(0, bval))
            if fill > 0:
                pygame.draw.rect(screen, bcolor, (track_x, y_cursor+2, fill, 14))
                screen.blit(alpha_surface(WHITE, fill, 7, 40), (track_x, y_cursor+2))
            screen.blit(font_mono_sm.render(f"{int(bval*100):3d}%", True, bcolor),
                        (track_x + track_w + 8, y_cursor))
            y_cursor += 26

        pygame.draw.line(screen, PANEL_BORDER, (px+20, y_cursor+4), (px+pw-20, y_cursor+4), 1)
        y_cursor += 16

    # ── Findings ──────────────────────────────────────────────────────────────
    if has_findings:
        screen.blit(font_mono_sm.render(
            f"CAPTURE-SYNC CORRELATION  ·  {findings.get('difficulty','')}",
            True, WHITE_DIM), (px+20, y_cursor))
        y_cursor += 20

        effect = findings["sync_effect"]
        cap_avg = findings["capture_avg_sync"]
        non_avg = findings["noncap_avg_sync"]
        interp  = findings["interpretation"]
        ic = (SYNCED_COLOR if "POSITIVE" in interp else DANGER if "NEGATIVE" in interp else WHITE_DIM)

        items = [
            ("SYNC AT CAPTURES",      f"{int(cap_avg*100)}%", SYNCED_COLOR),
            ("SYNC WITHOUT CAPTURES", f"{int(non_avg*100)}%", WHITE_DIM),
            ("SYNC EFFECT (Δ)",       f"{effect*100:+.1f}%",  ic),
            ("RESULT",                interp,                  ic),
        ]
        cw = pw // 4
        for i, (label, val, color) in enumerate(items):
            sx = px + i * cw + 20
            screen.blit(font_mono_sm.render(label, True, WHITE_DIM), (sx, y_cursor))
            screen.blit(font_mono_lg.render(val,   True, color),     (sx, y_cursor + 16))

        y_cursor += 58

    # ── Sync ring on right ────────────────────────────────────────────────────
    draw_sync_ring(px + pw - 90, py + ph - 90,
                   coherence_data['composite'] if coherence_data else avg_sync, tick)

    # ── CSV path notice ───────────────────────────────────────────────────────
    if findings and findings.get("csv_path"):
        csv_name = findings["csv_path"].split("/")[-1].split("\\")[-1]
        notice = font_mono_sm.render(f"CSV EXPORTED  →  {csv_name}", True, (55,75,115))
        screen.blit(notice, (px + 20, py + ph - 22))


def _draw_end_charts(screen, sync_timeline, capture_log, findings, avg_sync, tick):
    """Tab 1 — native pygame timeline chart + bar chart side by side."""
    margin = 20
    header_h = 54
    footer_h = 30
    usable_h = HEIGHT - header_h - footer_h - margin * 2

    # Left panel: timeline chart
    tl_x  = margin + 50          # extra left margin for y-axis labels
    tl_y  = header_h + margin
    tl_w  = int(WIDTH * 0.64) - 50 - margin
    tl_h  = usable_h

    # Right panel: bar chart
    bar_panel_x = tl_x + tl_w + 40
    bar_panel_w = WIDTH - bar_panel_x - margin
    bar_h_inner = usable_h - 60   # leave room for labels beneath
    bar_y       = tl_y + 30

    # Titles
    tl_title = font_mono_sm.render(
        "NEURAL SYNC TIMELINE  ·  CAPTURE EVENTS OVERLAID", True, CYAN_DIM)
    screen.blit(tl_title, (tl_x, tl_y - 18))

    bar_title = font_mono_sm.render("CAPTURE-SYNC CORRELATION", True, CYAN_DIM)
    screen.blit(bar_title, (bar_panel_x, tl_y - 18))

    draw_timeline_chart(tl_x, tl_y, tl_w, tl_h,
                        sync_timeline, capture_log, findings)

    draw_bar_chart(bar_panel_x, bar_y, bar_panel_w, bar_h_inner, findings)

    # Sync ring bottom-right
    draw_sync_ring(bar_panel_x + bar_panel_w//2,
                   bar_y + bar_h_inner + 45,
                   avg_sync, tick)


# ─── Difficulty Select Screen ─────────────────────────────────────────────────
DIFFICULTIES = {
    "SLOW":   {"speed": 1.8, "label": "SLOW",   "color": SYNCED_COLOR, "key": pygame.K_1, "desc": "Relaxed pace  ·  ideal for baseline"},
    "MEDIUM": {"speed": 3.0, "label": "MEDIUM",  "color": AMBER,        "key": pygame.K_2, "desc": "Standard difficulty"},
    "FAST":   {"speed": 5.5, "label": "FAST",    "color": DANGER,       "key": pygame.K_3, "desc": "High demand  ·  tests sync under stress"},
}

def draw_difficulty_screen(tick):
    screen.fill(BG_DARK)
    draw_grid(0.3)
    screen.blit(scanline_surf, (0, 0))

    title = font_mono_xl.render("NEURAL RESONANCE PROTOCOL", True, CYAN)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 140))
    sub = font_mono_md.render("SELECT DIFFICULTY  ·  PRESS  1 / 2 / 3", True, CYAN_DIM)
    screen.blit(sub, (WIDTH//2 - sub.get_width()//2, 204))

    card_w, card_h = 280, 160
    gap     = 40
    total_w = 3 * card_w + 2 * gap
    start_x = WIDTH//2 - total_w//2

    for i, (key, diff) in enumerate(DIFFICULTIES.items()):
        cx = start_x + i * (card_w + gap)
        cy = HEIGHT//2 - card_h//2

        card_bg = alpha_surface(BG_MID, card_w, card_h, 200)
        screen.blit(card_bg, (cx, cy))

        pulse        = 0.5 + 0.5 * math.sin(tick * 0.06 + i * 1.2)
        border_color = lerp_color(PANEL_BORDER, diff["color"], pulse * 0.4)
        pygame.draw.rect(screen, border_color, (cx, cy, card_w, card_h), 2)

        num = font_mono_xl.render(str(i+1), True, diff["color"])
        screen.blit(num, (cx + card_w//2 - num.get_width()//2, cy + 18))

        lbl = font_mono_lg.render(diff["label"], True, diff["color"])
        screen.blit(lbl, (cx + card_w//2 - lbl.get_width()//2, cy + 78))

        desc = font_mono_sm.render(diff["desc"], True, WHITE_DIM)
        screen.blit(desc, (cx + card_w//2 - desc.get_width()//2, cy + 118))

        for d in range(3):
            dot_color = diff["color"] if d <= i else PANEL_BORDER
            pygame.draw.circle(screen, dot_color, (cx + card_w//2 - 20 + d*20, cy + 144), 5)

    hint = font_mono_sm.render(
        "DIFFICULTY IS LOGGED AS AN INDEPENDENT VARIABLE IN YOUR RESEARCH EXPORT",
        True, (50, 70, 110))
    screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT - 80))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── UDP socket setup — done HERE, not at module level ─────────────────────
    # Moving bind() inside main() prevents an OSError on port 5005 (e.g. from a
    # previous run that hasn't fully released the socket) from crashing the
    # script before the pygame window ever opens and the difficulty screen shows.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow rapid restarts
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"[NET] Warning: could not bind UDP port {UDP_PORT}: {e}")
        print("[NET] Continuing without live BCI data (demo sine-wave mode).")
    sock.setblocking(False)

    player_pos   = [WIDTH // 2, HEIGHT // 2]
    target_pos   = [random.randint(120, WIDTH-120), random.randint(120, HEIGHT-160)]
    target_dir   = [random.choice([-1, 1]), random.choice([-1, 1])]
    target_speed = 3.0   # overwritten on difficulty select

    # ── State ─────────────────────────────────────────────────────────────────
    in_menu    = True
    game_over  = False
    difficulty = None
    time_limit = TIME_LIMIT
    start_time = None   # set when player picks difficulty

    score       = 0
    neural_sync = 0.0
    sync_history = []
    tick         = 0
    max_sync     = 0.0
    coherence_data = None
    band_vals    = {"alpha": 0.0, "beta": 0.0, "theta": 0.0}

    # ── Research logging ──────────────────────────────────────────────────────
    capture_log    = []
    sync_timeline  = []
    last_log_time  = 0.0
    findings       = None
    end_screen_tab = 0   # 0 = summary, 1 = charts

    running = True
    while running:
        tick += 1

        # ── Receive Brain Data (runs in all states so waveform is live on menu) ─
        try:
            data, addr = sock.recvfrom(1024)
            parts = data.decode().split(",")
            neural_sync = max(0.0, min(1.0, float(parts[0])))
            if len(parts) == 4:
                band_vals["alpha"] = float(parts[1])
                band_vals["beta"]  = float(parts[2])
                band_vals["theta"] = float(parts[3])
        except:
            # No data yet — use sine-wave demo signal
            neural_sync = 0.5 + 0.4 * math.sin(tick * 0.02)
            neural_sync = max(0.0, min(1.0, neural_sync))

        if not game_over:
            sync_history.append(neural_sync)
            if neural_sync > max_sync:
                max_sync = neural_sync

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # ── Tab switching on end screen ────────────────────────────
                if game_over:
                    key_char = getattr(event, 'unicode', '')
                    if event.key == pygame.K_1 or key_char == '1': end_screen_tab = 0
                    if event.key == pygame.K_2 or key_char == '2': end_screen_tab = 1
                    if event.key == pygame.K_TAB: end_screen_tab = 1 - end_screen_tab

                # ── Difficulty selection — only active in menu ─────────────
                # Check both pygame key constants AND event.unicode so the
                # selection works regardless of pygame version or keyboard layout.
                if in_menu:
                    choice = None
                    key_char = getattr(event, 'unicode', '')
                    if event.key == pygame.K_1 or key_char == '1': choice = "SLOW"
                    if event.key == pygame.K_2 or key_char == '2': choice = "MEDIUM"
                    if event.key == pygame.K_3 or key_char == '3': choice = "FAST"
                    # Also accept numpad
                    if event.key == pygame.K_KP1: choice = "SLOW"
                    if event.key == pygame.K_KP2: choice = "MEDIUM"
                    if event.key == pygame.K_KP3: choice = "FAST"
                    if choice:
                        difficulty   = DIFFICULTIES[choice]
                        target_speed = difficulty["speed"]
                        start_time   = time.time()
                        in_menu      = False

        # ── Menu render ───────────────────────────────────────────────────────
        if in_menu:
            draw_difficulty_screen(tick)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ── Timer ─────────────────────────────────────────────────────────────
        elapsed        = time.time() - start_time
        remaining_time = max(0, time_limit - elapsed)

        if remaining_time <= 0 and not game_over:
            game_over = True
            avg_sync_final = sum(sync_history)/len(sync_history) if sync_history else 0
            coherence_data = compute_session_coherence()
            findings = analyse_and_export(
                capture_log, sync_timeline, coherence_data,
                avg_sync_final, score, difficulty["label"])
            if coherence_data:
                print(f"\n[BCI] Session coherence: α={coherence_data['alpha']:.3f}  "
                      f"β={coherence_data['beta']:.3f}  θ={coherence_data['theta']:.3f}  "
                      f"composite={coherence_data['composite']:.3f}")

        # ── Game Over screen ──────────────────────────────────────────────────
        if game_over:
            avg_sync = sum(sync_history)/len(sync_history) if sync_history else 0
            draw_end_screen(screen, score, avg_sync, max_sync, tick,
                            coherence_data, findings,
                            sync_timeline=sync_timeline,
                            capture_log=capture_log,
                            tab=end_screen_tab)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ── Player Movement ───────────────────────────────────────────────────
        keys  = pygame.key.get_pressed()
        speed = 5 + (neural_sync * 9)
        if keys[pygame.K_a]:    player_pos[0] -= speed
        if keys[pygame.K_d]:    player_pos[0] += speed
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

        # ── Research: log sync once per second ───────────────────────────────
        if elapsed - last_log_time >= 1.0:
            sync_timeline.append((round(elapsed, 2), round(neural_sync, 4)))
            last_log_time = elapsed

        # ── Capture Logic ─────────────────────────────────────────────────────
        dist = math.hypot(player_pos[0] - target_pos[0], player_pos[1] - target_pos[1])
        if dist < 45:
            score += 1
            capture_log.append((round(elapsed, 2), round(neural_sync, 4)))
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

        edge_vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        edge_vignette.fill((0, 0, 20, 60))
        pygame.draw.circle(edge_vignette, (0, 0, 0, 0), (WIDTH//2, HEIGHT//2), 500)
        screen.blit(edge_vignette, (0, 0))

        for p in particles:
            p.draw(screen)

        draw_target(int(target_pos[0]), int(target_pos[1]), neural_sync, tick)
        draw_player(int(player_pos[0]), int(player_pos[1]), neural_sync, tick)

        draw_top_hud(remaining_time, score, neural_sync, tick)
        draw_bottom_hud(neural_sync, sync_history, tick)
        draw_side_stats(neural_sync, score, tick, band_vals)

        screen.blit(scanline_surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    start_bci_processing()
    main()