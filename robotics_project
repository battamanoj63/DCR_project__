from vpython import *
import random
import math
import csv
import json

# ============================================================
# AUTONOMOUS SURFACE WATER VEHICLE  -  RESCUE FLEET + DISPATCH
# ------------------------------------------------------------
# What this program does
#   1. EIGHT rescue boats wait at eight different bases.
#   2. A NEW INCIDENT LOCATION is chosen at random every time you
#      run the program (set INCIDENT_FIXED to force one).
#   3. DISPATCH CENTRE: for every boat the whole trip to the incident
#      is simulated beforehand (same sea, same current/wind, same moving
#      ships, rocks, parked boats).  The boat with the shortest estimated
#      arrival time (ETA) is chosen as the fastest responder.
#   4. Only that boat sails to the incident (the others stay on standby).
#      Predicted ETA is compared with the real arrival time at the end,
#      and the result is compared with "just send the nearest base".
#
# Sea state: MODERATE / ROUGH / STORM  (default ROUGH)
#   strong current + wind push the boat, steep waves turn the bow and
#   slow it down.
#
# ML / optimisation parts
#   1. Sliding-window linear regression  -> predicts where each
#      moving obstacle will be (from noisy position readings)
#   2. Tiny neural network (MLP, online) -> learns the hidden
#      water-current + wind drift and cancels it (feed-forward)
#   3. PSO (Particle Swarm Optimisation) -> tunes the controller
#      gains before the mission (random bases + random incidents)
#   4. Dataset logger -> usv_dataset.csv + usv_ml_model.json
#
# Controls (click the 3D window first)
#   C = camera (overview / chase / follow)   M = ML on/off
#   P = pause   S = save dataset             F = show/hide flow lines
# ============================================================

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
QUALITY = "HIGH"          # "HIGH", "MEDIUM" or "LOW"  (lower if it runs slowly)
SEA_STATE = "ROUGH"       # "MODERATE", "ROUGH" or "STORM"
INCIDENT_FIXED = None     # None = random incident every run, or e.g. (30.0, -20.0)
PSO_ENABLED = True        # False = skip tuning (starts faster, uses DEFAULT_PARAMS)

_PRESETS = {
    #           mesh cells, update every n frames, foam, flow lines, wind streaks
    "HIGH":   (28, 1, 110, 170, 70),
    "MEDIUM": (20, 2, 70, 110, 45),
    "LOW":    (14, 2, 40, 60, 30),
}
MESH_N, MESH_SKIP, FOAM_COUNT, FLOW_COUNT, WIND_STREAKS = _PRESETS[QUALITY]

SEA_PRESETS = {
    "MODERATE": dict(speed=0.22, wind=8.0, cur=(0.030, 0.012), meander=0.020, zone=1.0,
                     waves=(0.42, 0.16, 0.07, 0.055), heel=0.10, noise=0.15,
                     tilt=5.0, normal=3.0, foam=1.0, yaw=0.00, loss=0.00),
    "ROUGH":    dict(speed=0.26, wind=14.0, cur=(0.045, 0.018), meander=0.030, zone=1.4,
                     waves=(0.85, 0.32, 0.14, 0.11), heel=0.16, noise=0.28,
                     tilt=3.0, normal=2.0, foam=1.5, yaw=0.15, loss=0.08),
    "STORM":    dict(speed=0.30, wind=20.0, cur=(0.065, 0.026), meander=0.045, zone=1.8,
                     waves=(1.15, 0.45, 0.19, 0.15), heel=0.22, noise=0.35,
                     tilt=2.2, normal=1.6, foam=2.0, yaw=0.30, loss=0.15),
}
_SEA = SEA_PRESETS[SEA_STATE]
FOAM_COUNT = int(FOAM_COUNT * _SEA["foam"])

PSO_PARTICLES = 12
PSO_ITERS = 10
PSO_TRAIN_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 109, 127, 139]   # random base+incident pairs
PSO_TEST_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]               # unseen pairs
DEFAULT_PARAMS = [3.5, 0.3, 0.25, 9.0, 40.0]       # used when PSO_ENABLED = False

# ------------------------------------------------------------
# PHYSICAL PARAMETERS  (set by the sea state)
# ------------------------------------------------------------
dt = 0.05
USV_SPEED = _SEA["speed"]
HULL_RADIUS = 1.7
WIND_MEAN = _SEA["wind"]            # average wind speed (m/s)
WIND_LEEWAY = 0.0022                # boat push per step for each 1 m/s of wind
CURRENT_BASE = _SEA["cur"]          # steady large-scale current (push per step)
MEANDER_AMP = _SEA["meander"]       # slow wavy current that exists everywhere
HEEL_GAIN = _SEA["heel"]            # how strongly wind heels (rolls) the boat
WAVE_YAW = _SEA["yaw"]              # waves turn the bow
SPEED_LOSS = _SEA["loss"]           # speed lost in waves
WAVE_SCALE = 1.0                    # updated every frame: stronger wind -> bigger waves
OBS_NOISE = _SEA["noise"]           # sensor noise on obstacle positions
DRIFT_NOISE = 0.004                 # sensor noise on drift measurement
DRIFT_SCALE = 12.0                  # scaling so the network outputs are ~1
PRED_HORIZON_FRAMES = 40            # horizon used to score the predictor
MAX_STEPS = 1500
ARRIVE_R = 3.0                      # arrived when this close to the incident

WAVE_HEIGHT_1, WAVE_HEIGHT_2, WAVE_HEIGHT_3, WAVE_HEIGHT_4 = _SEA["waves"]
WAVE_SUM = WAVE_HEIGHT_1 + WAVE_HEIGHT_2 + WAVE_HEIGHT_3 + WAVE_HEIGHT_4
TILT_GAIN = _SEA["tilt"]
NORMAL_GAIN = _SEA["normal"]

# ---- the eight rescue bases (name, x, z) -------------------
BASES = [("South-West", -42.0, -38.0),
         ("West", -42.0, 8.0),
         ("South", 0.0, -44.0),
         ("South-East", 44.0, -36.0),
         ("North", -10.0, 44.0),
         ("East", 44.0, 10.0),
         ("North-West", -42.0, 40.0),
         ("North-East", 40.0, 44.0)]
BASE_XY = [(b[1], b[2]) for b in BASES]

STATIC_DATA = [
    (-20, -5, 3.5),
    (5, 15, 4.0),
    (-5, -25, 3.5),
    (25, -5, 3.8),
    (-30, 25, 4.0),
]
STATICS = [[x, z, 0.0, 0.0, r] for x, z, r in STATIC_DATA]


def random_incident(rng):
    """A random incident position in open water: clear of the rocks and at least
    22 m away from every base."""
    for _ in range(1000):
        x = rng.uniform(-38, 38)
        z = rng.uniform(-38, 38)
        if any(math.hypot(x - sx, z - sz) < sr + 9 for sx, sz, sr in STATIC_DATA):
            continue
        if min(math.hypot(x - bx, z - bz) for bx, bz in BASE_XY) < 22:
            continue
        return (x, z)
    return (30.0, 30.0)


# the incident of THIS run (different every time you start the program)
if INCIDENT_FIXED is not None:
    GOAL = (float(INCIDENT_FIXED[0]), float(INCIDENT_FIXED[1]))
else:
    GOAL = random_incident(random.Random())

# current zones: (cx, cz, dir_x, dir_z, strength, radius)
_zone_raw = [
    (-20, -10, 1.0, 0.4, 0.050, 12),
    (15, 10, -0.7, 0.8, 0.060, 11),
    (-5, 30, 0.8, -0.4, 0.045, 10),
    (30, -25, -0.8, -0.3, 0.065, 10),
]
ZONES = []
for _cx, _cz, _dx, _dz, _s, _R in _zone_raw:
    _m = math.hypot(_dx, _dz)
    ZONES.append((_cx, _cz, _dx / _m, _dz / _m, _s * _SEA["zone"], _R))


# ============================================================
# WAVE FUNCTION  (returns height and slopes -> real normals)
# ============================================================

def wave(x, z, t):
    a1 = t * 1.15 + x * 0.055 + z * 0.085
    a2 = t * 2.0 + x * 0.12 - z * 0.07
    a3 = t * 3.2 + x * 0.28 + z * 0.21
    a4 = t * 1.7 - x * 0.08 + z * 0.13

    y = (WAVE_HEIGHT_1 * math.sin(a1) + WAVE_HEIGHT_2 * math.sin(a2)
         + WAVE_HEIGHT_3 * math.sin(a3) + WAVE_HEIGHT_4 * math.sin(a4))

    c1, c2, c3, c4 = math.cos(a1), math.cos(a2), math.cos(a3), math.cos(a4)
    dx = (WAVE_HEIGHT_1 * 0.055 * c1 + WAVE_HEIGHT_2 * 0.12 * c2
          + WAVE_HEIGHT_3 * 0.28 * c3 - WAVE_HEIGHT_4 * 0.08 * c4)
    dz = (WAVE_HEIGHT_1 * 0.085 * c1 - WAVE_HEIGHT_2 * 0.07 * c2
          + WAVE_HEIGHT_3 * 0.21 * c3 + WAVE_HEIGHT_4 * 0.13 * c4)
    return y * WAVE_SCALE, dx * WAVE_SCALE, dz * WAVE_SCALE


def water_height(x, z, t):
    return wave(x, z, t)[0]


# ============================================================
# CURRENT + WIND  (continuous fields - the robot does NOT know them)
# ============================================================

def current_at(x, z, t):
    """Water velocity (push per step) at any point - defined everywhere."""
    tide = 1.0 + 0.35 * math.sin(0.12 * t)                 # slow tide pulsing
    cx = CURRENT_BASE[0] * tide + MEANDER_AMP * math.sin(0.06 * z + 0.10 * t)
    cz = CURRENT_BASE[1] * tide + MEANDER_AMP * math.cos(0.06 * x - 0.10 * t)
    for zx, zz, dx, dz, s_, R in ZONES:                    # 4 smooth jet streams
        d = math.hypot(x - zx, z - zz)
        g = math.exp(-(d / (0.55 * R)) ** 2)
        cx += dx * s_ * g
        cz += dz * s_ * g
    return cx, cz


def wind_at(x, z, t):
    """Wind velocity in m/s: slowly turning direction + moving gust fronts."""
    ang = 0.7 + 0.45 * math.sin(0.07 * t) + 0.12 * math.sin(0.31 * t + 0.02 * x)
    gust = 0.5 * max(0.0, math.sin(0.45 * t - 0.05 * (0.8 * x + 0.6 * z)))
    speed = WIND_MEAN * (0.85 + 0.15 * math.sin(0.11 * t) + gust)
    return speed * math.cos(ang), speed * math.sin(ang)


def drift_at(x, z, t):
    """Total push on the boat per step = water current + wind leeway."""
    c = current_at(x, z, t)
    w = wind_at(x, z, t)
    return c[0] + WIND_LEEWAY * w[0], c[1] + WIND_LEEWAY * w[1]


# ============================================================
# SCENARIO GENERATOR  (shared by live run and simulations)
# ============================================================

def make_scenario(rng, goal, routes, n_cross):
    """Moving vessels: n_cross of them cross the routes start -> incident,
    plus 3 wanderers."""
    movers = []
    for kk in range(n_cross):
        st = routes[kk % len(routes)]
        gx = goal[0] - st[0]
        gz = goal[1] - st[1]
        glen = math.hypot(gx, gz)
        ux, uz = gx / glen, gz / glen          # unit vector start -> incident
        px, pz = -uz, ux                       # perpendicular

        for _try in range(200):
            along = rng.uniform(0.2, 0.8) * glen
            side = rng.choice([-1, 1]) * rng.uniform(6, 16)
            x = st[0] + ux * along + px * side
            z = st[1] + uz * along + pz * side
            if abs(x) > 42 or abs(z) > 42:
                continue
            if math.hypot(x - goal[0], z - goal[1]) < 12:
                continue
            if any(math.hypot(x - sx, z - sz) < sr + 4 for sx, sz, sr in STATIC_DATA):
                continue
            if any(math.hypot(x - a, z - b) < 10 for a, b in BASE_XY):
                continue
            break
        sp = rng.uniform(0.05, 0.12)
        vx = -px * (1 if side > 0 else -1) * sp + ux * rng.uniform(-0.03, 0.03)
        vz = -pz * (1 if side > 0 else -1) * sp + uz * rng.uniform(-0.03, 0.03)
        movers.append([x, z, vx, vz, rng.uniform(1.8, 3.0)])

    # 3 vessels wander randomly
    while len(movers) < n_cross + 3:
        x = rng.uniform(-35, 35)
        z = rng.uniform(-35, 35)
        if any(math.hypot(x - a, z - b) < 12 for a, b in BASE_XY):
            continue
        if math.hypot(x - goal[0], z - goal[1]) < 12:
            continue
        if any(math.hypot(x - sx, z - sz) < sr + 4 for sx, sz, sr in STATIC_DATA):
            continue
        sp = rng.uniform(0.04, 0.12)
        ang = rng.uniform(0, 2 * math.pi)
        movers.append([x, z, sp * math.cos(ang), sp * math.sin(ang),
                       rng.uniform(1.8, 3.0)])
    return movers


def move_movers(movers):
    for m in movers:
        m[0] += m[2]
        m[1] += m[3]
        if m[0] > 45:
            m[0] = 45
            m[2] = -m[2]
        if m[0] < -45:
            m[0] = -45
            m[2] = -m[2]
        if m[1] > 45:
            m[1] = 45
            m[3] = -m[3]
        if m[1] < -45:
            m[1] = -45
            m[3] = -m[3]


# ============================================================
# ML #1  -  SLIDING-WINDOW LINEAR REGRESSION (obstacle tracker)
# ============================================================

class VelocityTracker:
    """Fits x(k) = a + b*k and z(k) = c + d*k by least squares over the
    last n noisy readings. Gives a filtered position and a velocity."""

    def __init__(self, n=15):
        self.n = n
        self.k = []
        self.x = []
        self.z = []

    def update(self, k, x, z):
        self.k.append(k)
        self.x.append(x)
        self.z.append(z)
        if len(self.k) > self.n:
            self.k.pop(0)
            self.x.pop(0)
            self.z.pop(0)

    def estimate(self):
        n = len(self.k)
        if n < 4:
            return self.x[-1], self.z[-1], 0.0, 0.0
        mk = sum(self.k) / n
        mx = sum(self.x) / n
        mz = sum(self.z) / n
        var = sum((k - mk) ** 2 for k in self.k)
        if var < 1e-9:
            return mx, mz, 0.0, 0.0
        vx = sum((k - mk) * (x - mx) for k, x in zip(self.k, self.x)) / var
        vz = sum((k - mk) * (z - mz) for k, z in zip(self.k, self.z)) / var
        dk = self.k[-1] - mk
        return mx + vx * dk, mz + vz * dk, vx, vz


# ============================================================
# ML #2  -  TINY NEURAL NETWORK (online learning of the drift)
# ============================================================

class MLP:
    """8 inputs -> hidden (tanh) -> 2 outputs, trained by plain SGD."""

    def __init__(self, n_in, n_h, n_out, seed=7):
        r = random.Random(seed)
        self.W1 = [[r.uniform(-0.8, 0.8) for _ in range(n_in)] for _ in range(n_h)]
        self.b1 = [0.0] * n_h
        self.W2 = [[r.uniform(-0.5, 0.5) for _ in range(n_h)] for _ in range(n_out)]
        self.b2 = [0.0] * n_out

    def forward(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(self.W1, self.b1)]
        y = [sum(w * hi for w, hi in zip(row, h)) + b
             for row, b in zip(self.W2, self.b2)]
        return h, y

    def predict(self, x):
        return self.forward(x)[1]

    def train(self, x, target, lr):
        h, y = self.forward(x)
        err = [yi - ti for yi, ti in zip(y, target)]
        dh = [sum(err[k] * self.W2[k][j] for k in range(len(err))) * (1 - h[j] * h[j])
              for j in range(len(h))]
        for k in range(len(self.W2)):
            for j in range(len(h)):
                self.W2[k][j] -= lr * err[k] * h[j]
            self.b2[k] -= lr * err[k]
        for j in range(len(h)):
            for i in range(len(x)):
                self.W1[j][i] -= lr * dh[j] * x[i]
            self.b1[j] -= lr * dh[j]
        return sum(e * e for e in err) / len(err)

    def to_dict(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


def drift_features(x, z, t, wind):
    """Inputs: position, tide/meander clock, and the wind measured by the boat."""
    return [x / 50.0, z / 50.0,
            math.sin(0.12 * t), math.cos(0.12 * t),
            math.sin(0.10 * t), math.cos(0.10 * t),
            wind[0] / WIND_MEAN, wind[1] / WIND_MEAN]


# ============================================================
# CONTROLLER  (pure Python so PSO / dispatch can simulate it fast)
# ============================================================

def controller(px, pz, obs, drift, P, goal):
    dx = goal[0] - px
    dz = goal[1] - pz
    d = math.hypot(dx, dz)
    if d < 1e-6:
        d = 1e-6
    gx = dx / d
    gz = dz / d

    ax = 0.0
    az = 0.0
    nearest = 999.0
    safe = P["safe"]
    ka = P["k_avoid"]
    ks = P["k_swirl"]
    H = P["horizon"]

    for ox, oz, vx, vz, r in obs:
        if H > 0 and (vx != 0.0 or vz != 0.0):
            samples = ((ox, oz, 1.0), (ox + vx * H, oz + vz * H, 0.7))
        else:
            samples = ((ox, oz, 1.0),)

        first = True
        for sx, sz, w in samples:
            ex = px - sx
            ez = pz - sz
            dist = math.hypot(ex, ez)
            if first:
                nearest = min(nearest, dist - r)
                first = False
            zone = safe + r
            if 0.1 < dist < zone:
                s = (zone - dist) / zone
                ux = ex / dist
                uz = ez / dist
                # sideways component (stops head-on dead-locks)
                tx = -uz
                tz = ux
                if tx * gx + tz * gz < 0:
                    tx = -tx
                    tz = -tz
                f = s * w * ka
                ax += (ux + ks * tx) * f
                az += (uz + ks * tz) * f

    ff = P.get("ff", 1.0)
    nx = gx + ax - ff * drift[0] / USV_SPEED
    nz = gz + az - ff * drift[1] / USV_SPEED
    n = math.hypot(nx, nz)
    if n < 1e-9:
        nx, nz = gx, gz
    else:
        nx /= n
        nz /= n
    return nx, nz, nearest, ax, az


def find_collision(x, z, obs):
    for o in obs:
        if math.hypot(x - o[0], z - o[1]) < o[4] + HULL_RADIUS:
            return o
    return None


def usv_step(S, P, obs_ctrl, obs_true, drift_ctrl, drift_true, t=0.0):
    """One simulation step. Used by the PSO runs, the dispatch simulations
    AND the live run."""
    goal = S["goal"]
    nx, nz, nearest, ax, az = controller(S["x"], S["z"], obs_ctrl, drift_ctrl, P, goal)

    a = P["alpha"]
    hx = S["hx"] * (1 - a) + nx * a
    hz = S["hz"] * (1 - a) + nz * a
    n = math.hypot(hx, hz)
    if n > 1e-9:
        hx /= n
        hz /= n
    else:
        hx, hz = nx, nz

    # after a bump the boat "flees" for a few steps at full speed
    flee_active = S["flee"] > 0
    if flee_active:
        hx, hz = S["fx"], S["fz"]
        S["flee"] -= 1

    # waves turn the bow (yaw kick from the slope across the boat) ...
    _wy, wdx, wdz = wave(S["x"], S["z"], t)
    if WAVE_YAW > 0.0 and not flee_active:
        lat = -wdx * hz + wdz * hx
        ang = WAVE_YAW * lat
        c_, s_ = math.cos(ang), math.sin(ang)
        hx, hz = hx * c_ - hz * s_, hx * s_ + hz * c_
    # ... and slow it down when it climbs a wave
    slope_long = wdx * hx + wdz * hz
    wave_factor = 1.0
    if SPEED_LOSS > 0.0:
        wave_factor = max(0.5, 1.0 - SPEED_LOSS - 1.2 * max(0.0, slope_long))

    safe = P["safe"]
    if nearest >= safe or flee_active:
        speed_factor = 1.0
    else:
        speed_factor = max(0.4, nearest / safe)
    sp = USV_SPEED * speed_factor * wave_factor

    cx = S["x"] + hx * sp + drift_true[0]
    cz = S["z"] + hz * sp + drift_true[1]

    hit = find_collision(cx, cz, obs_true)
    if hit is not None:
        S["coll"] += 1
        ex = S["x"] - hit[0]
        ez = S["z"] - hit[1]
        dd = math.hypot(ex, ez)
        if dd > 0.1:
            ux = ex / dd
            uz = ez / dd
            # bump: push the boat back out to the edge of the obstacle
            clear_d = hit[4] + HULL_RADIUS + 0.02
            S["x"] = hit[0] + ux * clear_d
            S["z"] = hit[1] + uz * clear_d
            # flee direction = away from the obstacle + sideways toward the goal side
            gx = goal[0] - S["x"]
            gz = goal[1] - S["z"]
            tx, tz = -uz, ux
            if tx * gx + tz * gz < 0:
                tx, tz = -tx, -tz
            fx = ux * 0.7 + tx * 0.7
            fz = uz * 0.7 + tz * 0.7
            fn = math.hypot(fx, fz)
            S["fx"] = fx / fn
            S["fz"] = fz / fn
            S["flee"] = 12
            hx, hz = S["fx"], S["fz"]
    else:
        S["path"] += math.hypot(cx - S["x"], cz - S["z"])
        S["x"] = cx
        S["z"] = cz

    S["x"] = max(-48.0, min(48.0, S["x"]))
    S["z"] = max(-48.0, min(48.0, S["z"]))

    S["turn"] += 1.0 - (hx * S["hx"] + hz * S["hz"])
    S["clear"] += max(0.0, 3.0 - nearest)
    S["hx"] = hx
    S["hz"] = hz
    return nearest


def new_state(start, goal):
    dx = goal[0] - start[0]
    dz = goal[1] - start[1]
    n = math.hypot(dx, dz)
    return {"x": start[0], "z": start[1], "hx": dx / n, "hz": dz / n,
            "coll": 0, "path": 0.0, "turn": 0.0, "clear": 0.0,
            "flee": 0, "fx": 1.0, "fz": 0.0, "goal": goal}


# ============================================================
# DISPATCH SIMULATION  (digital twin of one boat's trip)
# ============================================================

def simulate_mission(P, start, goal, movers0, extra_obs, record=False):
    """Simulate one boat from start to the incident without graphics.
    movers0 = the moving vessels at t = 0 (they are copied, not changed)."""
    movers = [m[:] for m in movers0]
    S = new_state(start, goal)
    pts = []
    reached = False
    step = 0
    for step in range(MAX_STEPS):
        t = step * dt
        move_movers(movers)
        obs = STATICS + movers + extra_obs
        d = drift_at(S["x"], S["z"], t)
        usv_step(S, P, obs, obs, d, d, t)
        if record and step % 6 == 0:
            pts.append((S["x"], S["z"]))
        if math.hypot(goal[0] - S["x"], goal[1] - S["z"]) < ARRIVE_R:
            reached = True
            break
    if record:
        pts.append((S["x"], S["z"]))
    return {"reached": reached, "time": (step + 1) * dt, "path": S["path"],
            "coll": S["coll"], "pts": pts, "state": S}


def run_dispatch(P, goal, movers0):
    """Estimate the ETA of EVERY base and rank them; rank 1 = fastest responder."""
    table = []
    for i, (name, bx, bz) in enumerate(BASES):
        parked = [[ox, oz, 0.0, 0.0, HULL_RADIUS]
                  for j, (n_, ox, oz) in enumerate(BASES) if j != i]
        r = simulate_mission(P, (bx, bz), goal, movers0, parked, record=True)
        r["id"] = i
        r["name"] = name
        r["straight"] = math.hypot(goal[0] - bx, goal[1] - bz)
        r["eta"] = r["time"] if r["reached"] else 1.0e9
        r["score"] = r["eta"] + 0.5 * r["coll"]          # collisions make a route worse
        table.append(r)
    order = sorted(table, key=lambda r: r["score"])
    for rank, r in enumerate(order, 1):
        r["rank"] = rank
    return table, order


# ============================================================
# ML #3  -  PSO CONTROLLER OPTIMISATION
# ============================================================

PARAM_NAMES = ["k_avoid", "k_swirl", "alpha", "safe", "horizon"]
PARAM_LO = [0.5, 0.0, 0.18, 6.0, 0.0]
PARAM_HI = [6.0, 1.5, 0.40, 11.0, 60.0]
MANUAL_PARAMS = [3.0, 0.0, 0.14, 7.0, 0.0]   # the original hand-tuned values


def vec_to_params(vec):
    P = dict(zip(PARAM_NAMES, vec))
    P["ff"] = 1.0
    return P


def mission_for(seed, k):
    """A reproducible (base, incident) pair used for tuning the controller."""
    r = random.Random(seed * 7 + 13)
    _name, bx, bz = BASES[k % len(BASES)]
    return (bx, bz), random_incident(r)


def simulate_headless(P, seed, k=0):
    """One tuning mission (random base + random incident); returns a cost."""
    start, goal = mission_for(seed, k)
    rng = random.Random(seed)
    movers = make_scenario(rng, goal, [start], 4)
    S = new_state(start, goal)
    steps = 0
    reached = False
    for step in range(MAX_STEPS):
        steps = step
        t = step * dt
        move_movers(movers)
        obs = STATICS + movers
        dtrue = drift_at(S["x"], S["z"], t)
        usv_step(S, P, obs, obs, dtrue, dtrue, t)
        if math.hypot(goal[0] - S["x"], goal[1] - S["z"]) < ARRIVE_R:
            reached = True
            break

    cost = (S["path"] + 0.1 * steps + 60.0 * S["coll"]
            + 4.0 * S["clear"] + 8.0 * S["turn"])
    if not reached:
        cost += 400 + 8 * math.hypot(goal[0] - S["x"], goal[1] - S["z"])
    return cost


def mean_cost(vec, seeds):
    P = vec_to_params(vec)
    return sum(simulate_headless(P, s, k) for k, s in enumerate(seeds)) / len(seeds)


def run_pso(progress=None, seed=2024):
    rng = random.Random(seed)
    dim = len(PARAM_NAMES)

    X = [[rng.uniform(PARAM_LO[d], PARAM_HI[d]) for d in range(dim)]
         for _ in range(PSO_PARTICLES)]
    X[0] = MANUAL_PARAMS[:]                       # seed the swarm with the manual design
    X[1] = [3.0, 0.3, 0.14, 7.0, 30.0]            # and one design that uses prediction
    V = [[0.0] * dim for _ in range(PSO_PARTICLES)]

    pbest = [x[:] for x in X]
    pcost = [mean_cost(x, PSO_TRAIN_SEEDS) for x in X]
    manual_train_cost = pcost[0]

    g = min(range(PSO_PARTICLES), key=lambda i: pcost[i])
    gbest = pbest[g][:]
    gcost = pcost[g]
    history = [gcost]
    if progress:
        progress(0, gcost, manual_train_cost)

    w, c1, c2 = 0.6, 1.5, 1.5
    for it in range(1, PSO_ITERS + 1):
        for i in range(PSO_PARTICLES):
            for d in range(dim):
                r1 = rng.random()
                r2 = rng.random()
                V[i][d] = (w * V[i][d]
                           + c1 * r1 * (pbest[i][d] - X[i][d])
                           + c2 * r2 * (gbest[d] - X[i][d]))
                X[i][d] += V[i][d]
                X[i][d] = max(PARAM_LO[d], min(PARAM_HI[d], X[i][d]))
            c = mean_cost(X[i], PSO_TRAIN_SEEDS)
            if c < pcost[i]:
                pcost[i] = c
                pbest[i] = X[i][:]
                if c < gcost:
                    gcost = c
                    gbest = X[i][:]
        history.append(gcost)
        if progress:
            progress(it, gcost, manual_train_cost)

    manual_test = mean_cost(MANUAL_PARAMS, PSO_TEST_SEEDS)
    best_test = mean_cost(gbest, PSO_TEST_SEEDS)
    return gbest, history, manual_train_cost, manual_test, best_test


# ===== VISUALS ==============================================
# Everything below needs VPython (3D scene, objects, main loop)
# ============================================================

scene = canvas(
    title="Autonomous Rescue Fleet - fastest responder dispatch (%s sea)" % SEA_STATE.lower(),
    width=1400,
    height=800,
    background=vector(0.10, 0.22, 0.40) if SEA_STATE == "MODERATE" else vector(0.07, 0.13, 0.24)
)
scene.autoscale = False
scene.center = vector(0, 0, 0)
scene.range = 58
scene.forward = vector(-0.35, -0.62, -0.70)

# ---------- Lighting ----------
scene.lights = []
scene.ambient = color.gray(0.38 if SEA_STATE == "MODERATE" else 0.28)
distant_light(direction=vector(-0.45, 0.60, -0.65), color=color.gray(0.95))   # sun
distant_light(direction=vector(0.30, 0.40, 0.90), color=color.gray(0.45))     # fill
distant_light(direction=vector(0.80, 0.20, -0.20), color=vector(0.25, 0.30, 0.40))

# ---------- Sky objects ----------
_sun_dir = norm(vector(-0.45, 0.35, -1.0))
sun = sphere(pos=_sun_dir * 150, radius=9, color=vector(1.0, 0.93, 0.72), emissive=True)
sun_glow = sphere(pos=_sun_dir * 150, radius=15, color=vector(1.0, 0.80, 0.50),
                  opacity=0.25, emissive=True)

# ============================================================
# WATER  (real shaded mesh instead of wire lines)
# ============================================================

if SEA_STATE == "MODERATE":
    DEEP = vector(0.01, 0.17, 0.36)
    SHALLOW = vector(0.06, 0.48, 0.68)
    CREST = vector(0.80, 0.94, 1.0)
else:                                     # darker, greyer, whiter crests
    DEEP = vector(0.01, 0.11, 0.23)
    SHALLOW = vector(0.05, 0.33, 0.48)
    CREST = vector(0.90, 0.96, 1.0)


def lerp_color(c1, c2, s):
    s = max(0.0, min(1.0, s))
    return c1 * (1 - s) + c2 * s


def water_color(y):
    s = (y / (WAVE_SCALE * WAVE_SUM) + 1.0) / 2.0
    c = lerp_color(DEEP, SHALLOW, s)
    if s > 0.74:
        c = lerp_color(c, CREST, (s - 0.74) / 0.26 * 0.70)
    return c


# deep base below the surface (also reaches the islands)
water_base = box(pos=vector(0, -2.2 - WAVE_SUM, 0), size=vector(330, 2.4, 330),
                 color=vector(0.01, 0.11, 0.25))

MESH_HALF = 50
_step = 2.0 * MESH_HALF / MESH_N
mesh_nodes = []
_grid = []
for _i in range(MESH_N + 1):
    _row = []
    for _j in range(MESH_N + 1):
        _x = -MESH_HALF + _i * _step
        _z = -MESH_HALF + _j * _step
        _y, _dx, _dz = wave(_x, _z, 0)
        _v = vertex(pos=vector(_x, _y, _z),
                    normal=norm(vector(-_dx * NORMAL_GAIN, 1, -_dz * NORMAL_GAIN)),
                    color=water_color(_y), opacity=0.93, shininess=0.9)
        _row.append(_v)
        mesh_nodes.append((_v, _x, _z))
    _grid.append(_row)

for _i in range(MESH_N):
    for _j in range(MESH_N):
        quad(vs=[_grid[_i][_j], _grid[_i][_j + 1],
                 _grid[_i + 1][_j + 1], _grid[_i + 1][_j]])


def update_mesh(t):
    for v, x, z in mesh_nodes:
        y, dx, dz = wave(x, z, t)
        v.pos = vector(x, y, z)
        v.normal = norm(vector(-dx * NORMAL_GAIN, 1, -dz * NORMAL_GAIN))
        v.color = water_color(y)


# arena border
curve(pos=[vector(-50, 0.25, -50), vector(50, 0.25, -50), vector(50, 0.25, 50),
           vector(-50, 0.25, 50), vector(-50, 0.25, -50)],
      color=vector(0.85, 0.95, 1.0), radius=0.07)

# ---------- Foam sparkles ----------
foam = []
for _i in range(FOAM_COUNT):
    _x = random.uniform(-50, 50)
    _z = random.uniform(-50, 50)
    _f = sphere(pos=vector(_x, water_height(_x, _z, 0) + 0.12, _z),
                radius=random.uniform(0.04, 0.12), color=vector(0.85, 0.95, 1.0),
                opacity=random.uniform(0.25, 0.6))
    foam.append([_f, random.uniform(-0.025, 0.025),
                 random.uniform(-0.025, 0.025), random.uniform(0, 6.28)])

# ---------- Distant islands ----------
def make_island(x, z, w, h):
    cylinder(pos=vector(x, -1.3, z), axis=vector(0, 1.4, 0), radius=w * 0.8,
             color=vector(0.80, 0.72, 0.52))
    ellipsoid(pos=vector(x, h * 0.25, z), size=vector(w * 1.3, h, w * 1.1),
              color=vector(0.16, 0.42, 0.20))
    ellipsoid(pos=vector(x + w * 0.2, h * 0.35, z - w * 0.15),
              size=vector(w * 0.8, h * 1.1, w * 0.7), color=vector(0.12, 0.34, 0.16))


make_island(-78, -62, 22, 9)
make_island(84, -72, 26, 11)
make_island(-86, 64, 20, 8)

# ============================================================
# STATIC ROCKS  (+ rings that ride the waves)
# ============================================================

rock_rings = []          # (ring, x, z) - every ring here follows the water surface
for _x, _z, _r in STATIC_DATA:
    sphere(pos=vector(_x, 0.5, _z), radius=_r, texture=textures.rock)
    sphere(pos=vector(_x + _r * 0.7, 0.2, _z + _r * 0.3), radius=_r * 0.55,
           texture=textures.rock)
    sphere(pos=vector(_x - _r * 0.5, 0.1, _z - _r * 0.6), radius=_r * 0.45,
           texture=textures.rock)
    _ring = ring(pos=vector(_x, 0.1, _z), axis=vector(0, 1, 0), radius=_r * 1.08,
                 thickness=0.16, color=vector(0.90, 0.97, 1.0), opacity=0.55)
    rock_rings.append((_ring, _x, _z))

# ============================================================
# BOAT BUILDERS
# ============================================================

BOAT_COLORS = [vector(0.96, 0.45, 0.08), vector(0.90, 0.20, 0.65),
               vector(0.95, 0.85, 0.10), vector(0.20, 0.85, 0.55),
               vector(0.35, 0.55, 1.00), vector(0.95, 0.30, 0.25),
               vector(0.20, 0.80, 0.85), vector(0.65, 0.40, 0.95)]


def build_usv(accent):
    parts = []
    parts.append(box(pos=vector(0, 0, 0), size=vector(5.0, 0.8, 2.0),
                     color=vector(0.93, 0.93, 0.96)))
    parts.append(pyramid(pos=vector(2.5, 0, 0), size=vector(2.2, 0.8, 2.0),
                         color=accent))
    parts.append(box(pos=vector(-0.2, 0.44, 0), size=vector(4.2, 0.08, 1.7),
                     color=vector(0.25, 0.27, 0.30)))
    parts.append(box(pos=vector(-0.4, 1.05, 0), size=vector(1.9, 1.1, 1.5),
                     color=vector(0.15, 0.35, 0.45)))
    parts.append(box(pos=vector(0.2, 1.2, 0), size=vector(0.9, 0.45, 1.56),
                     color=vector(0.05, 0.08, 0.10)))
    parts.append(box(pos=vector(-0.4, 1.65, 0), size=vector(2.0, 0.10, 1.6),
                     color=vector(0.95, 0.95, 0.95)))
    parts.append(cylinder(pos=vector(-0.8, 1.7, 0), axis=vector(0, 0.9, 0),
                          radius=0.10, color=color.white))
    parts.append(cylinder(pos=vector(-0.8, 2.55, 0), axis=vector(0, 0.18, 0),
                          radius=0.42, color=color.gray(0.2)))
    parts.append(sphere(pos=vector(-0.8, 2.75, 0), radius=0.22, color=accent,
                        emissive=True))
    parts.append(cylinder(pos=vector(-1.4, 1.7, 0.4), axis=vector(0, 1.1, 0),
                          radius=0.04, color=color.black))
    for zz in (-1.3, 1.3):
        parts.append(cylinder(pos=vector(-2.3, -0.15, zz), axis=vector(4.6, 0, 0),
                              radius=0.30, color=accent))
        parts.append(sphere(pos=vector(2.3, -0.15, zz), radius=0.30,
                            color=accent))
    parts.append(sphere(pos=vector(2.0, 0.55, -0.85), radius=0.12, color=color.red,
                        emissive=True))
    parts.append(sphere(pos=vector(2.0, 0.55, 0.85), radius=0.12, color=color.green,
                        emissive=True))
    return compound(parts)


def build_vessel(scale, col):
    L = scale * 2.6
    W = scale * 1.0
    H = scale * 0.5
    parts = [
        box(pos=vector(0, 0, 0), size=vector(L, H, W), color=col),
        pyramid(pos=vector(L / 2, 0, 0), size=vector(L * 0.35, H, W), color=col),
        box(pos=vector(-L * 0.1, H * 0.5 + scale * 0.3, 0),
            size=vector(L * 0.3, scale * 0.6, W * 0.7), color=color.white),
        cylinder(pos=vector(-L * 0.1, H * 0.5 + scale * 0.6, 0),
                 axis=vector(0, scale * 0.9, 0), radius=scale * 0.05,
                 color=color.gray(0.6)),
        sphere(pos=vector(-L * 0.1, H * 0.5 + scale * 1.5, 0), radius=scale * 0.12,
               color=color.red, emissive=True),
    ]
    return compound(parts)


def place_compound(obj, c0, P, f, u):
    """Put the compound so its local origin sits at P, facing f with up u."""
    r = cross(f, u)
    obj.pos = P + f * c0.x + u * c0.y + r * c0.z
    obj.axis = f
    obj.up = u


# ============================================================
# DESTINATION (the incident of this run)
# ============================================================

destination_position = vector(GOAL[0], 1.0, GOAL[1])

destination = sphere(pos=destination_position, radius=1.6, color=color.green,
                     emissive=True)
cylinder(pos=vector(GOAL[0], -0.6, GOAL[1]), axis=vector(0, 1.7, 0), radius=1.4,
         color=vector(0.95, 0.95, 0.95))
cylinder(pos=vector(GOAL[0], 0.0, GOAL[1]), axis=vector(0, 0.45, 0), radius=1.42,
         color=color.red)
destination_ring = ring(pos=destination_position, axis=vector(0, 1, 0), radius=3.5,
                        thickness=0.22, color=color.green)
beacon_beam = cylinder(pos=vector(GOAL[0], 2.0, GOAL[1]), axis=vector(0, 14, 0),
                       radius=0.35, color=color.green, opacity=0.20, emissive=True)
beacon_light = local_light(pos=vector(GOAL[0], 4, GOAL[1]), color=vector(0.1, 0.6, 0.2))

label(pos=destination_position + vector(0, 17, 0),
      text="EMERGENCY ASSISTANCE\nincident at (%.0f, %.0f)" % (GOAL[0], GOAL[1]),
      height=15, color=color.green, box=False)

# ============================================================
# DYNAMIC OBSTACLES  (vessels) + ML tracker + prediction ghost
# ============================================================

_rng_live = random.Random()
live_movers = make_scenario(_rng_live, GOAL, _rng_live.sample(BASE_XY, 5), 5)
movers_t0 = [m[:] for m in live_movers]          # what the dispatch centre sees at t = 0

_vessel_colors = [vector(0.90, 0.16, 0.10), vector(0.95, 0.45, 0.08),
                  vector(0.85, 0.12, 0.30), vector(0.95, 0.70, 0.10)]


class VesselVisual:
    pass


vessels = []
for _i, _m in enumerate(live_movers):
    vv = VesselVisual()
    _scale = _m[4] / 1.2
    vv.body = build_vessel(_scale, _vessel_colors[_i % len(_vessel_colors)])
    vv.c0 = vector(vv.body.pos.x, vv.body.pos.y, vv.body.pos.z)
    vv.tracker = VelocityTracker(15)
    vv.ghost = sphere(pos=vector(_m[0], 1.0, _m[1]), radius=_m[4], color=color.orange,
                      opacity=0.16)
    vv.line = curve(pos=[vector(_m[0], 1.0, _m[1]), vector(_m[0], 1.0, _m[1])],
                    color=color.orange, radius=0.07)
    vv.pending = []
    vessels.append(vv)

# ============================================================
# CURRENT ZONES + WIND (visuals only; physics is drift_at())
# ============================================================

class CurrentZoneVisual:

    def __init__(self, cx, cz, dx, dz, radius):
        self.pos = vector(cx, 0.9, cz)
        self.direction = vector(dx, 0, dz)
        self.radius = radius
        self.phase = random.uniform(0, 2 * math.pi)
        self.arrows = []
        self.boundary = ring(pos=self.pos, axis=vector(0, 1, 0), radius=radius,
                             thickness=0.12, color=color.cyan, opacity=0.25)
        for _ in range(8):
            ang = random.uniform(0, 2 * math.pi)
            rr = random.uniform(0.2, radius * 0.8)
            off = vector(math.cos(ang) * rr, 0, math.sin(ang) * rr)
            a = arrow(pos=self.pos + off, axis=self.direction * 2, shaftwidth=0.12,
                      headwidth=0.32, color=color.cyan, opacity=0.65)
            self.arrows.append((a, off, random.uniform(0, 2 * math.pi)))

    def update(self, t):
        for a, off, ph in self.arrows:
            a.pos = self.pos + off + self.direction * (math.sin(t * 1.4 + ph) * 1.5)
            a.axis = self.direction * (1.7 + 0.5 * math.sin(t * 2.5 + ph))
        self.boundary.radius = self.radius + 0.35 * math.sin(t * 2 + self.phase)


zone_visuals = [CurrentZoneVisual(cx, cz, dx, dz, R) for cx, cz, dx, dz, s, R in ZONES]

# ---------- Continuous flow lines (water) and wind streaks (air) ----------
FLOW_VIS = 4.0        # visual speed-up so slow currents are easy to see
FLOW_STREAK = 45.0    # streak length per unit of current speed
WIND_VIS = 0.02       # visual movement of wind streaks per m/s
FLOW_SLOW = vector(0.15, 0.45, 0.85)
FLOW_FAST = vector(0.80, 1.00, 1.00)

flow = []
for _i in range(FLOW_COUNT):
    _x = random.uniform(-48, 48)
    _z = random.uniform(-48, 48)
    _c = cylinder(pos=vector(_x, 0.3, _z), axis=vector(1.0, 0, 0), radius=0.07,
                  color=FLOW_SLOW, opacity=0.75)
    flow.append([_c, _x, _z, random.randint(40, 160)])

wind_streaks = []
for _i in range(WIND_STREAKS):
    _x = random.uniform(-55, 55)
    _z = random.uniform(-55, 55)
    _y = random.uniform(2.5, 6.5)
    _c = cylinder(pos=vector(_x, _y, _z), axis=vector(2, 0, 0), radius=0.05,
                  color=color.white, opacity=0.35)
    wind_streaks.append([_c, _x, _y, _z])

show_flow = True


def set_flow_visible(flag):
    for item in flow:
        item[0].visible = flag
    for item in wind_streaks:
        item[0].visible = flag


# ============================================================
# THE RESCUE FLEET: 8 boats waiting at their bases
# ============================================================

class Boat:
    pass


boats = []
for _i, (_name, _bx, _bz) in enumerate(BASES):
    b = Boat()
    b.id = _i
    b.name = _name
    b.col = BOAT_COLORS[_i % len(BOAT_COLORS)]
    b.S = new_state((_bx, _bz), GOAL)             # heading already points at the incident
    b.body = build_usv(b.col)
    b.c0 = vector(b.body.pos.x, b.body.pos.y, b.body.pos.z)
    b.tag = label(pos=vector(_bx, 6, _bz), text="%d %s" % (_i + 1, _name),
                  height=11, color=b.col, box=False)
    b.route = curve(color=b.col * 0.55, radius=0.05)
    b.info = None
    boats.append(b)

    _r1 = ring(pos=vector(_bx, 0.1, _bz), axis=vector(0, 1, 0), radius=3.0,
               thickness=0.22, color=b.col, opacity=0.9)
    rock_rings.append((_r1, _bx, _bz))

# ---------- effects of the dispatched boat (created now, used after dispatch) ----------
WAKE_N = 48
wake = [sphere(pos=vector(0, -5, 0), radius=0.1, color=vector(0.88, 0.96, 1.0),
               opacity=0.0) for _ in range(WAKE_N)]
wake_life = [0.0] * WAKE_N
wake_idx = 0


def emit_wake(P, f):
    global wake_idx
    s = wake[wake_idx]
    s.pos = P - f * 2.7 + vector(random.uniform(-0.4, 0.4), 0, random.uniform(-0.4, 0.4))
    wake_life[wake_idx] = 1.0
    wake_idx = (wake_idx + 1) % WAKE_N


def update_wake(t):
    for i, s in enumerate(wake):
        life = wake_life[i]
        if life > 0:
            life = max(0.0, life - 0.025)
            wake_life[i] = life
            s.radius = 0.15 + (1 - life) * 0.55
            s.opacity = life * 0.55
            s.pos.y = water_height(s.pos.x, s.pos.z, t) + 0.1


trail = curve(color=color.yellow, radius=0.08)
nav_arrow = arrow(pos=vector(0, 2, 0), axis=vector(5, 0, 0),
                  shaftwidth=0.18, headwidth=0.5, color=color.yellow, opacity=0.8)
sel_ring = ring(pos=vector(0, 0.2, 0), axis=vector(0, 1, 0), radius=4.0,
                thickness=0.25, color=color.yellow, opacity=0.9)

# arrows that show HOW the environment acts on the boat
wind_arrow = arrow(pos=vector(0, 5, 0), axis=vector(3, 0, 0),
                   shaftwidth=0.20, headwidth=0.55, color=color.white)
current_arrow = arrow(pos=vector(0, 1, 0), axis=vector(3, 0, 0),
                      shaftwidth=0.20, headwidth=0.55, color=color.cyan)
drift_arrow = arrow(pos=vector(0, 3, 0), axis=vector(3, 0, 0),
                    shaftwidth=0.22, headwidth=0.6, color=color.red)
cog_arrow = arrow(pos=vector(0, 2.2, 0), axis=vector(3, 0, 0),
                  shaftwidth=0.18, headwidth=0.5, color=color.green, opacity=0.9)

sonar_rings = []


def create_sonar(pos):
    sonar_rings.append(ring(pos=pos, axis=vector(0, 1, 0), radius=1, thickness=0.07,
                            color=color.cyan, opacity=0.65))


def place_usv(body, c0, x, z, hx, hz, t):
    y, dx, dz = wave(x, z, t)
    n = norm(vector(-dx * TILT_GAIN, 1, -dz * TILT_GAIN))
    f = vector(hx, 0, hz)
    f = norm(f - n * dot(f, n))

    # wind heels the boat: the mast leans toward the side the wind pushes
    w = wind_at(x, z, t)
    right = cross(f, n)
    lat = dot(vector(w[0], 0, w[1]), right) / WIND_MEAN
    u = norm(n + right * (lat * HEEL_GAIN))
    f = norm(f - u * dot(f, u))

    P = vector(x, y + 0.55, z)
    place_compound(body, c0, P, f, u)
    return P, f, y


# ============================================================
# HUD + CAMERA + KEYS
# ============================================================

status = label(pos=vector(33, 17, 0), text="INITIALIZING...", height=12,
               color=color.white, box=True, opacity=0.72, align="left")

ml_on = True
paused = False
cam_mode = 0
cam_center = vector(0, 0, 0)
cam_fwd = vector(-0.35, -0.62, -0.70)

CAM_NAMES = ["Overview", "Chase", "Follow"]


def set_camera_mode(m):
    global cam_mode, cam_center, cam_fwd
    cam_mode = m % 3
    cam_center = vector(scene.center.x, scene.center.y, scene.center.z)
    cam_fwd = norm(vector(scene.forward.x, scene.forward.y, scene.forward.z))
    if cam_mode == 0:
        scene.center = vector(0, 0, 0)
        scene.range = 58
        scene.forward = vector(-0.35, -0.62, -0.70)
        cam_center = vector(0, 0, 0)
    elif cam_mode == 1:
        scene.range = 14
    else:
        scene.range = 30


def on_key(evt):
    global ml_on, paused, show_flow
    k = evt.key.lower()
    if k == "f":
        show_flow = not show_flow
        set_flow_visible(show_flow)
    elif k == "c":
        set_camera_mode(cam_mode + 1)
    elif k == "m":
        ml_on = not ml_on
    elif k == "p":
        paused = not paused
    elif k == "s":
        save_outputs()


scene.bind("keydown", on_key)

# ============================================================
# ML #4  -  DATASET LOGGER
# ============================================================

dataset = []
DATA_COLS = ["time", "boat", "x", "z", "heading_x", "heading_z", "goal_dist",
             "nearest_obstacle", "drift_true_x", "drift_true_z", "drift_pred_x",
             "drift_pred_z", "ml_on", "collisions"]

best_params = None
pso_info = {}
dispatch_info = {}
drift_net = MLP(8, 16, 2)


def save_outputs():
    try:
        with open("usv_dataset.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(DATA_COLS)
            wr.writerows(dataset)
        with open("usv_ml_model.json", "w") as fh:
            json.dump({"sea_state": SEA_STATE,
                       "incident": GOAL,
                       "dispatch": dispatch_info,
                       "controller_params": best_params,
                       "pso": pso_info,
                       "drift_network": drift_net.to_dict()}, fh, indent=2)
        print("Saved usv_dataset.csv (%d rows) and usv_ml_model.json" % len(dataset))
    except Exception as e:
        print("Could not save files:", e)


# ============================================================
# STEP 1 :  OPTIMISE THE CONTROLLER WITH PSO  (random bases + incidents)
# ============================================================

def pso_progress(it, gcost, manual_cost):
    status.text = ("OPTIMIZING CONTROLLER (PSO)\n"
                   "----------------------\n"
                   f"Sea state : {SEA_STATE}\n"
                   f"Iteration : {it}/{PSO_ITERS}\n"
                   f"Manual cost : {manual_cost:.1f}\n"
                   f"Best cost : {gcost:.1f}")
    rate(60)


_mtest = 0.0
_btest = 0.0
if PSO_ENABLED:
    status.text = "OPTIMIZING CONTROLLER (PSO)...\nPlease wait a few seconds"
    rate(30)
    _g, _hist, _mtrain, _mtest, _btest = run_pso(pso_progress)
    best_params = vec_to_params(_g)
    pso_info = {"history": _hist, "manual_test_cost": _mtest, "optimised_test_cost": _btest,
                "manual_params": dict(zip(PARAM_NAMES, MANUAL_PARAMS))}
    print("PSO finished (%s sea)" % SEA_STATE)
    print("  tuned parameters :", {k: round(v, 3) for k, v in best_params.items()})
    print("  manual cost (unseen scenarios)    : %.1f" % _mtest)
    print("  optimised cost (unseen scenarios) : %.1f" % _btest)
else:
    best_params = vec_to_params(DEFAULT_PARAMS)
    pso_info = {"note": "PSO skipped, default parameters used"}

# ============================================================
# STEP 2 :  DISPATCH CENTRE - who is the fastest responder?
# ============================================================

status.text = "DISPATCH CENTRE\nSimulating the trip of all %d boats..." % len(BASES)
rate(30)

dispatch_table, dispatch_order = run_dispatch(best_params, GOAL, movers_t0)
ab = boats[dispatch_order[0]["id"]]                  # the active (dispatched) boat
fastest = dispatch_order[0]
nearest_base = min(dispatch_table, key=lambda r: r["straight"])

print("\nINCIDENT at (%.1f, %.1f)   sea: %s" % (GOAL[0], GOAL[1], SEA_STATE))
print("DISPATCH TABLE (estimated arrival time of every boat)")
print("  rank  boat  base          straight(m)  ETA(s)  path(m)  hits")
for r in dispatch_order:
    eta_txt = "%6.1f" % r["eta"] if r["reached"] else "  none"
    print("  %2d    %d     %-12s  %9.1f   %s  %7.1f  %4d"
          % (r["rank"], r["id"] + 1, r["name"], r["straight"], eta_txt, r["path"], r["coll"]))
print("=> FASTEST RESPONDER: boat %d (%s), predicted ETA %.1f s" %
      (fastest["id"] + 1, fastest["name"], fastest["eta"]))
if nearest_base["id"] != fastest["id"]:
    print("   (the NEAREST base was boat %d (%s): ETA %.1f s - dispatch saved %.1f s)"
          % (nearest_base["id"] + 1, nearest_base["name"], nearest_base["eta"],
             nearest_base["eta"] - fastest["eta"]))
else:
    print("   (the fastest responder is also the nearest base)")

dispatch_info = {"fastest": fastest["id"] + 1, "nearest_base": nearest_base["id"] + 1,
                 "table": [{"boat": r["id"] + 1, "base": r["name"], "rank": r["rank"],
                            "straight_m": round(r["straight"], 1),
                            "eta_s": round(r["eta"], 1) if r["reached"] else None,
                            "collisions": r["coll"]} for r in dispatch_order]}

# --- draw the predicted routes, label every boat with its rank and ETA ---
for r in dispatch_table:
    bt = boats[r["id"]]
    for (px_, pz_) in r["pts"]:
        bt.route.append(pos=vector(px_, 0.9, pz_))
    eta_txt = "ETA %.0f s" % r["eta"] if r["reached"] else "no route"
    bt.tag.text = "%d %s\n#%d  %s" % (bt.id + 1, bt.name, r["rank"], eta_txt)
    bt.info = r
ab.tag.text = "%d %s\nDISPATCHED  ETA %.0f s" % (ab.id + 1, ab.name, fastest["eta"])
ab.tag.color = color.yellow
ab.tag.height = 14
ab.route.color = color.yellow
ab.route.radius = 0.12

dispatch_lines = ""
for r in dispatch_order:
    dispatch_lines += "%d. %-11s %4.0f m  %s%s\n" % (
        r["rank"], r["name"], r["straight"],
        ("%5.0f s" % r["eta"]) if r["reached"] else "  none",
        "  <- SENT" if r["id"] == ab.id else "")

# other (parked) boats are obstacles for the dispatched one
parked = [[o.S["x"], o.S["z"], 0.0, 0.0, HULL_RADIUS] for o in boats if o is not ab]

# ============================================================
# STATE
# ============================================================

S = ab.S
mission_complete = False
simulation_time = 0.0
frame_no = 0
sonar_timer = 0.0

replay = []
train_count = 0
loss_ema = 1.0
drift_err_ml = 0.02
drift_err_none = 0.02

pred_sq_ml = 0.0
pred_sq_static = 0.0
pred_n = 0

caption_timer = 0
t_arrival = None

mvx = 0.0
mvz = 0.0
cur_speed = 0.0
wind_speed = 0.0
drift_speed = 0.0
crab_deg = 0.0
sog = 0.0


def update_caption(nearest_true):
    rm = math.sqrt(pred_sq_ml / pred_n) if pred_n else 0.0
    rs = math.sqrt(pred_sq_static / pred_n) if pred_n else 0.0
    gain = 0.0
    if _mtest > 0:
        gain = 100.0 * (_mtest - _btest) / _mtest
    scene.caption = (
        "Keys:  [C] camera (%s)   [M] ML on/off   [P] pause   [S] save dataset   [F] flow lines\n"
        "Sea state : %s | incident at (%.0f, %.0f) | dispatched boat %d (%s) | ML assist : %s\n"
        "Environment : current %.2f m/s | wind %.1f m/s | wave scale %.2f | "
        "net push on boat %.2f m/s | crab angle %+.1f deg\n"
        "Arrows at the boat : yellow = heading, green = real course, cyan = current, "
        "white = wind, red = net push\n"
        "Controller (PSO) : cost %.1f -> %.1f on unseen scenarios (%.0f%% better) | "
        "gains k_avoid=%.2f swirl=%.2f alpha=%.2f safe=%.1f horizon=%.0f\n"
        "Obstacle predictor (regression) : RMSE %.2f m vs %.2f m if ignoring motion  (%d samples)\n"
        "Drift network (MLP) : loss %.4f | drift error %.4f learned vs %.4f ignored | "
        "trained %d steps\n"
        "Dataset rows : %d"
        % (CAM_NAMES[cam_mode], SEA_STATE, GOAL[0], GOAL[1], ab.id + 1, ab.name,
           "ON" if ml_on else "OFF (baseline)",
           cur_speed, wind_speed, WAVE_SCALE, drift_speed, crab_deg,
           _mtest, _btest, gain,
           best_params["k_avoid"], best_params["k_swirl"], best_params["alpha"],
           best_params["safe"], best_params["horizon"],
           rm, rs, pred_n, loss_ema, drift_err_ml, drift_err_none, train_count,
           len(dataset)))


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    rate(30)

    if paused:
        continue

    frame_no += 1
    simulation_time += dt
    t = simulation_time

    # ---------------- ocean ----------------
    if frame_no % MESH_SKIP == 0:
        update_mesh(t)

    for item in foam:
        f = item[0]
        cf = current_at(f.pos.x, f.pos.z, t)
        wf = wind_at(f.pos.x, f.pos.z, t)
        f.pos.x += cf[0] * FLOW_VIS + wf[0] * 0.004
        f.pos.z += cf[1] * FLOW_VIS + wf[1] * 0.004
        if f.pos.x > 52:
            f.pos.x = -52
        if f.pos.x < -52:
            f.pos.x = 52
        if f.pos.z > 52:
            f.pos.z = -52
        if f.pos.z < -52:
            f.pos.z = 52
        f.pos.y = water_height(f.pos.x, f.pos.z, t) + 0.12
        f.radius = 0.07 + 0.035 * math.sin(t * 2 + item[3])

    for ringobj, rx, rz in rock_rings:
        ringobj.pos.y = water_height(rx, rz, t) + 0.08

    for zv in zone_visuals:
        zv.update(t)

    # wind makes the sea rougher
    _w0 = wind_at(0, 0, t)
    WAVE_SCALE = 0.7 + 0.3 * math.hypot(_w0[0], _w0[1]) / WIND_MEAN

    # water flow lines follow the current field
    if show_flow:
        for k, item in enumerate(flow):
            c = item[0]
            vx_, vz_ = current_at(item[1], item[2], t)
            item[1] += vx_ * FLOW_VIS
            item[2] += vz_ * FLOW_VIS
            item[3] -= 1
            if item[3] <= 0 or abs(item[1]) > 50 or abs(item[2]) > 50:
                item[1] = random.uniform(-48, 48)
                item[2] = random.uniform(-48, 48)
                item[3] = random.randint(60, 160)
            c.pos = vector(item[1], water_height(item[1], item[2], t) + 0.25, item[2])
            c.axis = vector(vx_, 0, vz_) * FLOW_STREAK
            if (frame_no + k) % 12 == 0:
                c.color = lerp_color(FLOW_SLOW, FLOW_FAST, math.hypot(vx_, vz_) / 0.16)

        # wind streaks follow the wind field
        for item in wind_streaks:
            wx_, wz_ = wind_at(item[1], item[3], t)
            item[1] += wx_ * WIND_VIS
            item[3] += wz_ * WIND_VIS
            if item[1] > 56:
                item[1] -= 112
            if item[1] < -56:
                item[1] += 112
            if item[3] > 56:
                item[3] -= 112
            if item[3] < -56:
                item[3] += 112
            item[0].pos = vector(item[1], item[2], item[3])
            item[0].axis = vector(wx_, 0, wz_) * 0.45

    # ---------------- moving obstacles + ML tracker ----------------
    move_movers(live_movers)
    obs_true = STATICS + live_movers + parked     # what really exists
    obs_ctrl = list(STATICS) + parked             # what the boat "sees"

    for i, m in enumerate(live_movers):
        vv = vessels[i]

        # draw vessel (rides waves, faces its direction of travel)
        y, dx, dz = wave(m[0], m[1], t)
        n = norm(vector(-dx * TILT_GAIN, 1, -dz * TILT_GAIN))
        fvec = vector(m[2], 0, m[3])
        if mag(fvec) < 1e-6:
            fvec = vector(1, 0, 0)
        fvec = norm(fvec - n * dot(fvec, n))
        place_compound(vv.body, vv.c0, vector(m[0], y + 0.45, m[1]), fvec, n)

        # noisy sensor reading -> regression
        ox = m[0] + random.gauss(0, OBS_NOISE)
        oz = m[1] + random.gauss(0, OBS_NOISE)
        vv.tracker.update(frame_no, ox, oz)
        ex, ez, vx, vz = vv.tracker.estimate()

        if ml_on:
            obs_ctrl.append([ex, ez, vx, vz, m[4]])
        else:
            obs_ctrl.append([ox, oz, 0.0, 0.0, m[4]])

        # predictor scoring (ML vs "obstacle will stay put")
        if frame_no % 5 == 0 and len(vv.tracker.k) >= 4:
            vv.pending.append((frame_no + PRED_HORIZON_FRAMES,
                               ex + vx * PRED_HORIZON_FRAMES,
                               ez + vz * PRED_HORIZON_FRAMES, ex, ez))
        still = []
        for tgt, px_, pz_, x0, z0 in vv.pending:
            if frame_no >= tgt:
                pred_sq_ml += (m[0] - px_) ** 2 + (m[1] - pz_) ** 2
                pred_sq_static += (m[0] - x0) ** 2 + (m[1] - z0) ** 2
                pred_n += 1
            else:
                still.append((tgt, px_, pz_, x0, z0))
        vv.pending = still

        # prediction visuals
        H = best_params["horizon"] if best_params["horizon"] > 5 else 30
        px_v = ex + vx * H
        pz_v = ez + vz * H
        vv.ghost.visible = ml_on
        vv.line.visible = ml_on
        if ml_on:
            vv.ghost.pos = vector(px_v, water_height(px_v, pz_v, t) + 0.8, pz_v)
            vv.line.modify(0, pos=vector(m[0], y + 0.9, m[1]))
            vv.line.modify(1, pos=vector(px_v, water_height(px_v, pz_v, t) + 0.9, pz_v))

    # ---------------- standby boats ride the waves at their bases ----------------
    for bt in boats:
        if bt is ab:
            continue
        Pb, _fb, _yb = place_usv(bt.body, bt.c0, bt.S["x"], bt.S["z"],
                                 bt.S["hx"], bt.S["hz"], t)
        bt.tag.pos = vector(Pb.x, Pb.y + 6.0, Pb.z)

    # ---------------- drift network (ML) ----------------
    d_true = drift_at(S["x"], S["z"], t)
    wind_true = wind_at(S["x"], S["z"], t)
    wind_meas = (wind_true[0] + random.gauss(0, 0.3), wind_true[1] + random.gauss(0, 0.3))
    feats = drift_features(S["x"], S["z"], t, wind_meas)
    out = drift_net.predict(feats)
    d_pred = (out[0] / DRIFT_SCALE, out[1] / DRIFT_SCALE)
    conf = min(1.0, train_count / 500.0)
    if ml_on:
        drift_ctrl = (d_pred[0] * conf, d_pred[1] * conf)
    else:
        drift_ctrl = (0.0, 0.0)

    # ---------------- navigation of the dispatched boat ----------------
    target_distance = math.hypot(GOAL[0] - S["x"], GOAL[1] - S["z"])
    nearest_true = 999.0
    for o in obs_true:
        nearest_true = min(nearest_true, math.hypot(S["x"] - o[0], S["z"] - o[1]) - o[4])

    x_prev = S["x"]
    z_prev = S["z"]
    mvx = 0.0
    mvz = 0.0

    if not mission_complete:
        usv_step(S, best_params, obs_ctrl, obs_true, drift_ctrl, d_true, t)
        mvx = S["x"] - x_prev
        mvz = S["z"] - z_prev

        # learn drift from a noisy residual measurement
        meas = [(d_true[0] + random.gauss(0, DRIFT_NOISE)) * DRIFT_SCALE,
                (d_true[1] + random.gauss(0, DRIFT_NOISE)) * DRIFT_SCALE]
        sample = (feats, meas)
        if len(replay) < 600:
            replay.append(sample)
        else:
            replay[random.randrange(600)] = sample
        batch = [sample] + random.sample(replay, min(6, len(replay)))
        loss = 0.0
        for fx, tx in batch:
            loss += drift_net.train(fx, tx, 0.03)
        loss /= len(batch)
        loss_ema = 0.98 * loss_ema + 0.02 * loss
        train_count += 1
        drift_err_ml = 0.98 * drift_err_ml + 0.02 * math.hypot(d_pred[0] - d_true[0],
                                                                  d_pred[1] - d_true[1])
        drift_err_none = 0.98 * drift_err_none + 0.02 * math.hypot(d_true[0], d_true[1])

        if frame_no % 5 == 0:
            dataset.append([round(t, 2), ab.id + 1, round(S["x"], 3), round(S["z"], 3),
                            round(S["hx"], 3), round(S["hz"], 3),
                            round(target_distance, 3), round(nearest_true, 3),
                            round(d_true[0], 5), round(d_true[1], 5),
                            round(d_pred[0], 5), round(d_pred[1], 5),
                            int(ml_on), S["coll"]])

    # ---------------- draw the dispatched USV ----------------
    P, f, ocean_y = place_usv(ab.body, ab.c0, S["x"], S["z"], S["hx"], S["hz"], t)
    nav_arrow.pos = vector(P.x, P.y + 1.5, P.z)
    nav_arrow.axis = vector(S["hx"], 0, S["hz"]) * 5
    sel_ring.pos = vector(P.x, ocean_y + 0.2, P.z)
    ab.tag.pos = vector(P.x, P.y + 6.0, P.z)

    # ---- how current + wind act on the boat ----
    cur_now = current_at(S["x"], S["z"], t)
    wnd_now = wind_at(S["x"], S["z"], t)
    push_now = drift_at(S["x"], S["z"], t)
    cur_speed = math.hypot(cur_now[0], cur_now[1]) / dt
    wind_speed = math.hypot(wnd_now[0], wnd_now[1])
    drift_speed = math.hypot(push_now[0], push_now[1]) / dt
    sog = math.hypot(mvx, mvz) / dt
    if sog > 0.3:
        crab_deg = math.degrees(math.atan2(mvz, mvx) - math.atan2(S["hz"], S["hx"]))
        crab_deg = (crab_deg + 180.0) % 360.0 - 180.0

    wind_arrow.pos = vector(P.x, P.y + 4.2, P.z)
    wind_arrow.axis = vector(wnd_now[0], 0, wnd_now[1]) * (3.2 / WIND_MEAN)
    current_arrow.pos = vector(P.x, P.y + 0.7, P.z)
    current_arrow.axis = vector(cur_now[0], 0, cur_now[1]) * 45
    drift_arrow.pos = vector(P.x, P.y + 3.0, P.z)
    drift_arrow.axis = vector(push_now[0], 0, push_now[1]) * 60
    cog_arrow.pos = vector(P.x, P.y + 2.2, P.z)
    if mvx * mvx + mvz * mvz > 1e-8:
        cog_arrow.axis = vector(mvx, 0, mvz) * 25
    else:
        cog_arrow.axis = vector(0.01, 0, 0)

    if not mission_complete:
        if frame_no % 2 == 0:
            emit_wake(P, f)
            trail.append(pos=vector(P.x, ocean_y + 0.3, P.z))

        sonar_timer += dt
        if sonar_timer > 1.0:
            create_sonar(vector(P.x, ocean_y + 0.3, P.z))
            sonar_timer = 0.0

    update_wake(t)

    for sr in sonar_rings[:]:
        sr.pos = vector(P.x, ocean_y + 0.3, P.z)
        sr.radius += 0.35
        sr.opacity -= 0.025
        if sr.opacity <= 0:
            sr.visible = False
            sonar_rings.remove(sr)

    # ---------------- destination check ----------------
    if (not mission_complete) and target_distance < ARRIVE_R:
        mission_complete = True
        t_arrival = t
        nav_arrow.visible = False
        label(pos=P + vector(0, 7, 0), text="EMERGENCY ASSISTANCE REACHED", height=20,
              color=color.green, box=True, opacity=0.85)
        print("\nBoat %d (%s) reached the incident at t = %.1f s (predicted %.1f s), "
              "path %.1f m, collisions %d"
              % (ab.id + 1, ab.name, t, fastest["eta"], S["path"], S["coll"]))
        if nearest_base["id"] != fastest["id"] and nearest_base["reached"]:
            print("Sending the nearest base instead would have taken about %.1f s."
                  % nearest_base["eta"])
        save_outputs()

    # ---------------- camera ----------------
    if cam_mode == 1:
        cam_center = cam_center * 0.88 + vector(P.x, 1.5, P.z) * 0.12
        cam_fwd = norm(cam_fwd * 0.94 + norm(vector(S["hx"], -0.30, S["hz"])) * 0.06)
        scene.center = cam_center
        scene.forward = cam_fwd
    elif cam_mode == 2:
        cam_center = cam_center * 0.9 + vector(P.x, 1.0, P.z) * 0.1
        scene.center = cam_center

    # ---------------- HUD ----------------
    head = ("RESCUE FLEET DISPATCH (%s sea)\n"
            "Incident (%.0f, %.0f)\n"
            "rank base          dist    ETA\n" % (SEA_STATE, GOAL[0], GOAL[1]))
    if mission_complete:
        tail = ("----------------------\n"
                "MISSION COMPLETED\n"
                f"Boat {ab.id + 1} ({ab.name}) on scene\n"
                f"Arrival {t_arrival:.1f} s (predicted {fastest['eta']:.1f} s)\n"
                f"Path {S['path']:.1f} m | Collisions {S['coll']}")
    else:
        obstacle_status = "AVOIDING" if nearest_true < best_params["safe"] else "CLEAR"
        tail = ("----------------------\n"
                f"Boat {ab.id + 1} ({ab.name}) : {obstacle_status}\n"
                f"Target distance : {target_distance:.1f} m\n"
                f"Nearest obstacle : {nearest_true:.1f} m\n"
                f"Current {cur_speed:.2f} | Wind {wind_speed:.1f} m/s\n"
                f"Net push {drift_speed:.2f} m/s | Crab {crab_deg:+.1f} deg\n"
                f"Speed over ground : {sog:.2f} m/s\n"
                f"Collisions : {S['coll']} | Time : {simulation_time:.1f} s")
    status.text = head + dispatch_lines + tail

    caption_timer += 1
    if caption_timer >= 8:
        caption_timer = 0
        update_caption(nearest_true)

    # ---------------- destination animation ----------------
    destination_ring.radius = 3.5 + 0.7 * math.sin(t * 3)
    dy = water_height(GOAL[0], GOAL[1], t) + 1.0
    destination.pos.y = dy
    destination_ring.pos.y = dy
    beacon_beam.opacity = 0.16 + 0.08 * math.sin(t * 4)