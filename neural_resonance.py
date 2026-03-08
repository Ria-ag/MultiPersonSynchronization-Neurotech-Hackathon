import pygame
import random, sys
import socket
import time, math
import threading
from scipy.signal import hilbert

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

                phaseA = np.angle(hilbert(A))
                phaseB = np.angle(hilbert(B))

                phase_diff = phaseA - phaseB
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                return plv

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

                alpha_syncs = []
                beta_syncs = []
                theta_syncs = []

                for ch in range(4):
                    alpha_syncs.append(band_sync(personA[ch], personB[ch], 8, 12))
                    beta_syncs.append(band_sync(personA[ch], personB[ch], 13, 30))
                    theta_syncs.append(band_sync(personA[ch], personB[ch], 4, 7))

                alpha_sync = np.mean(alpha_syncs)
                beta_sync  = np.mean(beta_syncs)
                theta_sync = np.mean(theta_syncs)

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
                
                if not hasattr(run, "log"):
                    run.log = open("session_log.csv","w")
                    run.log.write("time,alpha,beta,theta,score\n")

                t = time.time()

                run.log.write(f"{t},{alpha_sync},{beta_sync},{theta_sync},{team_score_clamped}\n")
                run.log.flush()

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
pygame.display.set_caption("NEURAL SYNCHRONICITY TEST  ·  OpenBCI")
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
    title = font_mono_md.render("NEURAL SYNCHRONICITY TEST", True, CYAN)
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

    # ── Build Plotly HTML ─────────────────────────────────────────────────────
    try:
        import json
        times      = [t for t, _ in sync_timeline]
        syncs      = [s for _, s in sync_timeline]
        cap_t      = [t for t, _ in capture_log]
        cap_s      = [s for _, s in capture_log]

        # Rolling avg (5-point)
        def rolling(arr, w=5):
            out = []
            for i in range(len(arr)):
                sl = arr[max(0, i-w):i+w+1]
                out.append(sum(sl)/len(sl))
            return out
        smooth = rolling(syncs)

        # Coherence breakdown for annotation
        coh_text = ""
        if coherence_data:
            coh_text = (f"α={int(coherence_data['alpha']*100)}%  "
                        f"β={int(coherence_data['beta']*100)}%  "
                        f"θ={int(coherence_data['theta']*100)}%  "
                        f"composite={int(coherence_data['composite']*100)}%")

        effect_str = (f"{findings['sync_effect']*100:+.1f}%"
                      if findings['sync_effect'] is not None else "N/A")

        cap_avg_pct  = int((findings['capture_avg_sync'] or 0) * 100)
        non_avg_pct  = int((findings['noncap_avg_sync']  or 0) * 100)
        avg_sync_pct = int(avg_sync * 100)
        effect_val   = findings['sync_effect'] or 0
        interp_str   = findings['interpretation']
        diff_color   = 'positive' if difficulty == 'SLOW' else 'amber' if difficulty == 'MEDIUM' else 'negative'
        effect_color = 'positive' if effect_val > 0.03 else 'negative' if effect_val < -0.03 else 'neutral'
        interp_color_cls = 'positive' if 'POSITIVE' in interp_str else 'negative' if 'NEGATIVE' in interp_str else 'neutral'

        # Inline Plotly data
        data = {
            "sync_times":  times,
            "syncs":       syncs,
            "smooth":      smooth,
            "cap_t":       cap_t,
            "cap_s":       cap_s,
            "difficulty":  difficulty,
            "avg_sync":    round(avg_sync, 3),
            "cap_avg":     findings["capture_avg_sync"],
            "non_avg":     findings["noncap_avg_sync"],
            "effect":      findings["sync_effect"],
            "effect_str":  effect_str,
            "interp":      findings["interpretation"],
            "coh_text":    coh_text,
            "timestamp":   timestamp,
            "score":       score,
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Neural Resonance — {difficulty} — {timestamp}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ background: #04060e; color: #dce8ff; font-family: 'Courier New', monospace;
         margin: 0; padding: 30px; }}
  h1   {{ color: #00dcff; letter-spacing: 4px; font-size: 1.3em; margin-bottom: 4px; }}
  .sub {{ color: #006490; font-size: 0.8em; letter-spacing: 2px; margin-bottom: 30px; }}
  .stats {{ display: flex; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }}
  .stat  {{ background: #080c1c; border: 1px solid #1e3264;
            padding: 14px 22px; min-width: 160px; }}
  .stat .label {{ color: #506090; font-size: 0.72em; letter-spacing: 2px; }}
  .stat .value {{ font-size: 1.5em; margin-top: 4px; }}
  .positive {{ color: #50ffa0; }}
  .neutral  {{ color: #dce8ff; }}
  .negative {{ color: #ff3c3c; }}
  .amber    {{ color: #ffb400; }}
</style>
</head>
<body>
<h1>NEURAL SYNCHRONICITY TEST</h1>
<div class="sub">INTER-BRAIN SYNCHRONIZATION REPORT  ·  DIFFICULTY: {difficulty}  ·  {timestamp}</div>

<div class="stats">
  <div class="stat">
    <div class="label">DIFFICULTY</div>
    <div class="value {'positive' if difficulty=='SLOW' else 'amber' if difficulty=='MEDIUM' else 'negative'}">{difficulty}</div>
  </div>
  <div class="stat">
    <div class="label">AVG NEURAL SYNC</div>
    <div class="value neutral">{int(avg_sync*100)}%</div>
  </div>
  <div class="stat">
    <div class="label">TOTAL CAPTURES</div>
    <div class="value amber">{score}</div>
  </div>
  <div class="stat">
    <div class="label">SYNC AT CAPTURES</div>
    <div class="value positive">{cap_avg_pct}%</div>
  </div>
  <div class="stat">
    <div class="label">SYNC WITHOUT CAPTURES</div>
    <div class="value neutral">{non_avg_pct}%</div>
  </div>
  <div class="stat">
    <div class="label">SYNC EFFECT (Δ)</div>
    <div class="value {effect_color}">{effect_str}</div>
  </div>
  <div class="stat">
    <div class="label">FINDING</div>
    <div class="value {interp_color_cls}" style="font-size:0.9em">{interp_str}</div>
  </div>
</div>

<div id="chart"></div>
<div id="bars" style="margin-top:20px"></div>

<script>
const d = {json.dumps(data)};

// ── Main chart: sync timeline + capture markers ────────────────────────────
const syncLine = {{
  x: d.sync_times, y: d.syncs,
  type: 'scatter', mode: 'lines', name: 'Raw Sync',
  line: {{ color: 'rgba(0,200,255,0.35)', width: 1 }},
}};
const smoothLine = {{
  x: d.sync_times, y: d.smooth,
  type: 'scatter', mode: 'lines', name: 'Smoothed Sync',
  line: {{ color: '#00dcff', width: 2.5 }},
}};
const captures = {{
  x: d.cap_t, y: d.cap_s,
  type: 'scatter', mode: 'markers', name: 'Capture Event',
  marker: {{ color: '#ffb400', size: 12, symbol: 'diamond',
             line: {{ color: '#fff', width: 1 }} }},
}};

// Horizontal reference lines
const capAvgLine = {{
  x: [d.sync_times[0], d.sync_times[d.sync_times.length-1]],
  y: [d.cap_avg, d.cap_avg],
  type: 'scatter', mode: 'lines', name: 'Avg sync at captures',
  line: {{ color: 'rgba(80,255,160,0.6)', width: 1.5, dash: 'dot' }},
}};
const nonAvgLine = {{
  x: [d.sync_times[0], d.sync_times[d.sync_times.length-1]],
  y: [d.non_avg, d.non_avg],
  type: 'scatter', mode: 'lines', name: 'Avg sync without captures',
  line: {{ color: 'rgba(220,235,255,0.4)', width: 1.5, dash: 'dot' }},
}};

const layout1 = {{
  paper_bgcolor: '#04060e', plot_bgcolor: '#080c1c',
  font: {{ family: 'Courier New', color: '#dce8ff', size: 11 }},
  title: {{ text: 'NEURAL SYNC TIMELINE  ·  CAPTURE EVENTS OVERLAID',
            font: {{ color: '#00dcff', size: 13 }} }},
  xaxis: {{ title: 'Time (seconds)', gridcolor: '#0f1630', zeroline: false }},
  yaxis: {{ title: 'Neural Sync Score', range: [-0.05, 1.05],
            gridcolor: '#0f1630', zeroline: false }},
  legend: {{ bgcolor: '#080c1c', bordercolor: '#1e3264', borderwidth: 1 }},
  hovermode: 'x unified',
  annotations: [{{
    x: 0.01, y: 0.97, xref: 'paper', yref: 'paper',
    text: 'Δ = ' + d.effect_str + '  ·  ' + d.interp,
    showarrow: false,
    font: {{ color: d.effect > 0.03 ? '#50ffa0' : d.effect < -0.03 ? '#ff3c3c' : '#dce8ff',
             size: 12 }},
    align: 'left',
  }}],
}};

Plotly.newPlot('chart', [syncLine, smoothLine, captures, capAvgLine, nonAvgLine], layout1,
  {{responsive: true, displayModeBar: true}});

// ── Bar chart: capture vs non-capture avg sync ─────────────────────────────
const bars = {{
  x: ['Sync at Captures', 'Sync Without Captures'],
  y: [d.cap_avg, d.non_avg],
  type: 'bar',
  marker: {{ color: ['#50ffa0', 'rgba(100,120,160,0.6)'],
             line: {{ color: ['#50ffa0', '#506090'], width: 1.5 }} }},
  text: [Math.round(d.cap_avg*100)+'%', Math.round(d.non_avg*100)+'%'],
  textposition: 'outside',
  textfont: {{ color: '#dce8ff' }},
}};

const layout2 = {{
  paper_bgcolor: '#04060e', plot_bgcolor: '#080c1c',
  font: {{ family: 'Courier New', color: '#dce8ff', size: 11 }},
  title: {{ text: 'CAPTURE-SYNC CORRELATION  ·  KEY FINDING',
            font: {{ color: '#00dcff', size: 13 }} }},
  yaxis: {{ title: 'Avg Neural Sync', range: [0, 1.1],
            gridcolor: '#0f1630', zeroline: false }},
  height: 340,
}};

Plotly.newPlot('bars', [bars], layout2, {{responsive: true, displayModeBar: false}});
</script>
</body>
</html>"""

        with open(html_path, "w") as f:
            f.write(html)
        print(f"[RESEARCH] Graph exported → {html_path}")

    except Exception as e:
        import traceback
        print(f"[RESEARCH] HTML export failed: {e}")
        traceback.print_exc()

    print(f"[RESEARCH] CSV exported  → {csv_path}")
    print(f"[RESEARCH] Finding: {findings['interpretation']}  Δ={findings['sync_effect']}")
    return findings


# ─── End Screen ───────────────────────────────────────────────────────────────
def draw_end_screen(screen, score, avg_sync, max_sync, tick, coherence_data=None, findings=None):
    screen.fill(BG_DARK)
    draw_grid(avg_sync)
    screen.blit(scanline_surf, (0, 0))

    has_findings = findings and findings["sync_effect"] is not None
    pw = 800
    ph = 520 if has_findings else (460 if coherence_data else 400)
    px, py = WIDTH//2 - pw//2, HEIGHT//2 - ph//2
    panel = alpha_surface(BG_MID, pw, ph, 230)
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, PANEL_BORDER, (px, py, pw, ph), 2)
    pygame.draw.rect(screen, lerp_color(CYAN, SYNCED_COLOR, avg_sync), (px, py, pw, 3))

    title = font_mono_xl.render("SESSION COMPLETE", True, WHITE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, py + 22))

    sub = font_mono_md.render("INTER-BRAIN SYNCHRONIZATION REPORT", True, CYAN_DIM)
    screen.blit(sub, (WIDTH//2 - sub.get_width()//2, py + 76))
    pygame.draw.line(screen, PANEL_BORDER, (px + 40, py + 108), (px + pw - 40, py + 108), 1)

    # ── Top stats ─────────────────────────────────────────────────────────────
    coh_val   = f"{int(coherence_data['composite']*100)}%" if coherence_data else "N/A"
    coh_color = lerp_color(CYAN, SYNCED_COLOR, coherence_data['composite']) if coherence_data else WHITE_DIM

    stats = [
        ("NEURAL COHERENCE",  coh_val,               coh_color),
        ("AVG LIVE SYNC",     f"{int(avg_sync*100)}%", lerp_color(CYAN, WHITE, 0.5)),
        ("PEAK SYNC",         f"{int(max_sync*100)}%", SYNCED_COLOR),
        ("TOTAL CAPTURES",    str(score),              AMBER),
    ]
    col_w = (pw - 80) // 2
    for i, (label, val, color) in enumerate(stats):
        col = i % 2
        row = i // 2
        sx = px + 40 + col * (col_w + 40)
        sy = py + 122 + row * 75
        screen.blit(font_mono_sm.render(label, True, WHITE_DIM), (sx, sy))
        screen.blit(font_mono_lg.render(val, True, color),        (sx, sy + 18))

    y_cursor = py + 122 + 2 * 75 + 10

    # ── Coherence band breakdown ───────────────────────────────────────────────
    if coherence_data:
        pygame.draw.line(screen, PANEL_BORDER, (px+40, y_cursor), (px+pw-40, y_cursor), 1)
        y_cursor += 10
        screen.blit(font_mono_sm.render(
            f"COHERENCE BREAKDOWN  ·  {coherence_data['n_samples']} SAMPLES  ·  MAGNITUDE-SQUARED",
            True, WHITE_DIM), (px+40, y_cursor))
        y_cursor += 18
        bands = [
            ("α  ALPHA  8-12Hz", coherence_data['alpha'], CYAN),
            ("β  BETA  13-30Hz", coherence_data['beta'],  PINK),
            ("θ  THETA   4-7Hz", coherence_data['theta'], AMBER),
        ]
        track_x = px + 220
        track_w = pw - 280
        for blabel, bval, bcolor in bands:
            screen.blit(font_mono_sm.render(blabel, True, WHITE_DIM), (px+40, y_cursor))
            pygame.draw.rect(screen, (20,30,60),     (track_x, y_cursor+2, track_w, 13))
            pygame.draw.rect(screen, PANEL_BORDER,   (track_x, y_cursor+2, track_w, 13), 1)
            fill = int(track_w * max(0, bval))
            if fill > 0:
                pygame.draw.rect(screen, bcolor, (track_x, y_cursor+2, fill, 13))
                screen.blit(alpha_surface(WHITE, fill, 6, 40), (track_x, y_cursor+2))
            screen.blit(font_mono_sm.render(f"{int(bval*100):3d}%", True, bcolor),
                        (track_x + track_w + 8, y_cursor))
            y_cursor += 24

    # ── FINDINGS panel ────────────────────────────────────────────────────────
    if has_findings:
        y_cursor += 6
        pygame.draw.line(screen, PANEL_BORDER, (px+40, y_cursor), (px+pw-40, y_cursor), 1)
        y_cursor += 10

        # Header
        screen.blit(font_mono_sm.render(
            f"CAPTURE-SYNC CORRELATION  ·  KEY FINDING  ·  {findings.get('difficulty','')}", True, WHITE_DIM),
                    (px+40, y_cursor))
        y_cursor += 20

        effect     = findings["sync_effect"]
        cap_avg    = findings["capture_avg_sync"]
        non_avg    = findings["noncap_avg_sync"]
        interp     = findings["interpretation"]
        interp_color = (SYNCED_COLOR if "POSITIVE" in interp
                        else DANGER    if "NEGATIVE" in interp
                        else WHITE_DIM)

        # Two stat columns
        col_items = [
            ("SYNC AT CAPTURES",     f"{int(cap_avg*100)}%",  SYNCED_COLOR),
            ("SYNC WITHOUT CAPTURES",f"{int(non_avg*100)}%",  WHITE_DIM),
            ("SYNC EFFECT (Δ)",      f"{effect*100:+.1f}%",   interp_color),
            ("RESULT",               interp,                   interp_color),
        ]
        for i, (label, val, color) in enumerate(col_items):
            col = i % 2
            row = i // 2
            sx = px + 40 + col * (col_w + 40)
            sy = y_cursor + row * 50
            screen.blit(font_mono_sm.render(label, True, WHITE_DIM), (sx, sy))
            screen.blit(font_mono_md.render(val,   True, color),     (sx, sy + 16))

        y_cursor += 2 * 50 + 10

        # CSV export notice
        csv_name = findings["html_path"].split("/")[-1].split("\\")[-1]
        screen.blit(font_mono_sm.render(f"GRAPH EXPORTED  →  {csv_name}", True, (60,80,120)),
                    (px+40, y_cursor))
        y_cursor += 18

    # ── Footer ────────────────────────────────────────────────────────────────
    pygame.draw.line(screen, PANEL_BORDER, (px+40, y_cursor+4), (px+pw-40, y_cursor+4), 1)
    screen.blit(font_mono_sm.render("PRESS  ESC  TO  EXIT", True, WHITE_DIM),
                (WIDTH//2 - font_mono_sm.size("PRESS  ESC  TO  EXIT")[0]//2, y_cursor + 12))

    draw_sync_ring(WIDTH//2 + 360, HEIGHT//2,
                   coherence_data['composite'] if coherence_data else avg_sync, tick)


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

    title = font_mono_xl.render("NEURAL SYNCHRONICITY TEST", True, CYAN)
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
    capture_log   = []
    sync_timeline = []
    last_log_time = 0.0
    findings      = None

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
            draw_end_screen(screen, score, avg_sync, max_sync, tick, coherence_data, findings)
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