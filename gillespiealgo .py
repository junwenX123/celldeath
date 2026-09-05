import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# ============================================================
# Gillespie / event-driven simulation for the model,
# with a fixed T-shaped background zone.
#
# Spatial domain: W = [0, Lx] x [0, Ly].
#
# Core model:
#   - hidden caspase activation process V^a_t;
#   - observed death process V^d_t;
#   - ERK protection process V^p_t generated after accepted deaths.
#
# T-shaped extension:
#   outside active zones, the background activation intensity is not uniform:
#       inside the fixed T-shaped zone:     lambda_a_T
#       outside the fixed T-shaped zone:    lambda_a_c
#   inside active zones, the intensity is always lambda_a_1.
#
# Target activation intensity:
#   lambda_a(x | V^a_{t-}) =
#       lambda_a_1,  if x is in A(V^a_{t-}),
#       lambda_a_T,  if x is not in A(V^a_{t-}) but is in the fixed T-zone,
#       lambda_a_c,  otherwise.
#
# Simulation principle:
#   Everything is handled by one Gillespie total rate a0:
#       1. activation candidate proposal,
#       2. death candidate proposal,
#       3. active-center expiration,
#       4. ERK-protection-zone expiration.
#
# Important: in this version ERK expiration is NOT pre-sampled by erk_end_t.
# ERK zones expire through their Gillespie rate N^p_t * beta_d_T, exactly like
# active centers expire through N^a_t * beta_a_T.
# ============================================================


# -----------------------------
# 1. Parameters
# -----------------------------

seed = 123
rng = np.random.default_rng(seed)

# Spatial domain W = [0, Lx] x [0, Ly]
Lx = 10.0
Ly = 10.0
area_W = Lx * Ly

# Simulation stopping rule.
T_max = 20.0
target_deaths = None

# Activation intensities.
lambda_a_1 = 5.00   # dominating intensity and active-zone intensity
lambda_a_T = 0.50   # background activation intensity inside fixed T-zone
lambda_a_c = 0.05   # background activation intensity outside T-zone

if not (0.0 <= lambda_a_c <= lambda_a_T <= lambda_a_1):
    raise ValueError("Need 0 <= lambda_a_c <= lambda_a_T <= lambda_a_1 for thinning.")

# Candidate death intensity on W.
lambda_d = 1.0

# Activation marks: R^a ~ Exp(beta_a_R), lifetime clock rate beta_a_T.
beta_a_R = 2.5
beta_a_T = 1.2

# ERK marks: R^d ~ Exp(beta_d_R), lifetime clock rate beta_d_T.
# In this unified Gillespie version, T^d is represented by an expiration rate,
# not by storing a pre-sampled end time.
beta_d_R = 2.0
beta_d_T = 0.8

# Pause between visual updates. Set to 0.0 for faster runs.
display_pause = 0.008

# Draw rejected candidates or not.
show_rejected_points = False


# -----------------------------
# 2. Fixed T-shaped zone
# -----------------------------

x1 = Lx / 3.0
x2 = 2.0 * Lx / 3.0
y1 = Ly / 3.0
y2 = 2.0 * Ly / 3.0


def is_inside_T_zone(x, y):
    """Return True if point (x, y) is inside the fixed T-shaped region."""
    inside_middle_column = (x1 <= x <= x2) and (0.0 <= y <= Ly)
    inside_left_arm = (0.0 <= x <= x1) and (y1 <= y <= y2)
    return inside_middle_column or inside_left_arm


# -----------------------------
# 3. State variables
# -----------------------------

t = 0.0

# Active centers V^a_t = sum delta_(Y_i^a, R_i^a).
# These are the active centers that are alive at current time t.
active_x = []
active_y = []
active_r = []

# Accepted caspase activation events X = {(Y_i^a, S_i^a)}.
# These are historical accepted activation events.
cas_x = []
cas_y = []
cas_t = []
cas_origin = []

# Observed cell death events Y = {(Y_i^d, S_i^d)}.
death_x = []
death_y = []
death_t = []

# ERK protection zones V^p_t.
# These are active protection zones alive at current time t.
# In this unified Gillespie version, we do not store erk_start_t or erk_end_t.
erk_x = []
erk_y = []
erk_r = []

# Rejected candidate points, only used for optional display.
rej_x = []
rej_y = []
rej_t = []
rej_type = []

# Histories.
event_times = [0.0]
active_counts = [0]
cas_counts = [0]
death_counts = [0]
erk_counts = [0]


# -----------------------------
# 4. Helper functions
# -----------------------------

def is_inside_active_zone(x, y):
    """Return True if (x, y) belongs to A(V^a_t) = union_i B(Y_i^a, R_i^a)."""
    for cx, cy, r in zip(active_x, active_y, active_r):
        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
            return True
    return False


def local_activation_intensity(x, y):
    """
    Return the T-shaped extension of the activation intensity:
        lambda_a_1 inside A(V^a_t),
        lambda_a_T inside the fixed T-zone but outside A(V^a_t),
        lambda_a_c outside both.
    """
    if is_inside_active_zone(x, y):
        return lambda_a_1
    if is_inside_T_zone(x, y):
        return lambda_a_T
    return lambda_a_c


def activation_origin(x, y):
    """Return a readable label for the origin of an accepted activation event."""
    if is_inside_active_zone(x, y):
        return "active zone"
    if is_inside_T_zone(x, y):
        return "T-shaped background zone"
    return "background outside T"


def is_inside_erk_protection(x, y):
    """
    Return True if (x, y) is inside an active ERK protection zone.

    If True, a candidate death is rejected because the death intensity is 0 there.
    """
    for ex, ey, er in zip(erk_x, erk_y, erk_r):
        if (x - ex) ** 2 + (y - ey) ** 2 <= er ** 2:
            return True
    return False


def local_death_intensity(x, y):
    """Return lambda_d(x | V^a_t, V^p_t)."""
    if is_inside_active_zone(x, y) and not is_inside_erk_protection(x, y):
        return lambda_d
    return 0.0


def add_active_center(x, y):
    """Accepted activation creates a new active center with radius R^a ~ Exp(beta_a_R)."""
    r = rng.exponential(scale=1.0 / beta_a_R)
    active_x.append(x)
    active_y.append(y)
    active_r.append(r)
    return r


def remove_random_active_center():
    """
    Conditional on an active-center expiration event, every active center has
    the same exponential lifetime rate beta_a_T. Hence choose one uniformly.
    """
    n = len(active_x)
    if n == 0:
        return None

    idx = rng.integers(0, n)
    active_x.pop(idx)
    active_y.pop(idx)
    active_r.pop(idx)
    return idx


def add_erk_protection_zone(x, y):
    """
    After an accepted death, create an ERK protection zone.

    R^d ~ Exp(beta_d_R).
    The lifetime T^d ~ Exp(beta_d_T) is simulated by the Gillespie expiration
    rate len(erk_x) * beta_d_T, not by storing a sampled end time.
    """
    r_E = rng.exponential(scale=1.0 / beta_d_R)
    erk_x.append(x)
    erk_y.append(y)
    erk_r.append(r_E)
    return r_E


def remove_random_erk_zone():
    """
    Conditional on an ERK expiration event, every ERK protection zone has the
    same exponential lifetime rate beta_d_T. Hence choose one uniformly.
    """
    n = len(erk_x)
    if n == 0:
        return None

    idx = rng.integers(0, n)
    erk_x.pop(idx)
    erk_y.pop(idx)
    erk_r.pop(idx)
    return idx


def record_history():
    event_times.append(t)
    active_counts.append(len(active_x))
    cas_counts.append(len(cas_x))
    death_counts.append(len(death_x))
    erk_counts.append(len(erk_x))


# -----------------------------
# 5. Matplotlib real-time setup
# -----------------------------

plt.ion()

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Gillespie simulation: caspase + death + ERK, with T-shaped background")

# Draw fixed T-shaped lambda_a_T background zone.
t_patch_vertical = Rectangle(
    (x1, 0.0),
    x2 - x1,
    Ly,
    fill=True,
    alpha=0.12,
    linewidth=0.0,
    label=r"fixed $T$ zone: $\lambda^a_T$",
)
ax.add_patch(t_patch_vertical)

t_patch_arm = Rectangle(
    (0.0, y1),
    x1,
    y2 - y1,
    fill=True,
    alpha=0.12,
    linewidth=0.0,
)
ax.add_patch(t_patch_arm)

# Draw 3 x 3 grid.
for xx in [x1, x2]:
    ax.axvline(xx, linestyle="--", linewidth=0.8, alpha=0.35)
for yy in [y1, y2]:
    ax.axhline(yy, linestyle="--", linewidth=0.8, alpha=0.35)

# Background intensity labels.
ax.text(0.5 * x1, 0.5 * (y2 + Ly), r"$\lambda^a_c$", ha="center", va="center", fontsize=13, alpha=0.75)
ax.text(0.5 * x1, 0.5 * y1, r"$\lambda^a_c$", ha="center", va="center", fontsize=13, alpha=0.75)
ax.text(0.5 * (x2 + Lx), 0.5 * Ly, r"$\lambda^a_c$", ha="center", va="center", fontsize=13, alpha=0.75)
ax.text(0.5 * (x1 + x2), 0.5 * Ly, r"$\lambda^a_T$", ha="center", va="center", fontsize=14, alpha=0.75)

# Accepted activation events.
cas_scatter = ax.scatter([], [], s=18, label=r"accepted activations $X$")

# Observed death events.
death_scatter = ax.scatter([], [], s=55, marker="x", linewidths=2.0, label=r"observed deaths $Y$")

# Rejected candidate points, optional.
rejected_scatter = ax.scatter([], [], s=10, marker="x", alpha=0.30, label="rejected candidates")

# Active centers.
active_scatter = ax.scatter([], [], s=55, label=r"active centers $V^a_t$")

# ERK protection centers.
erk_scatter = ax.scatter([], [], s=25, alpha=0.45, label=r"ERK centers $V^p_t$")

time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
event_text = ax.text(0.02, 0.93, "", transform=ax.transAxes, va="top", ha="left")
count_text = ax.text(0.02, 0.86, "", transform=ax.transAxes, va="top", ha="left")
rate_text = ax.text(0.02, 0.81, "", transform=ax.transAxes, va="top", ha="left")
death_text = ax.text(0.02, 0.76, "", transform=ax.transAxes, va="top", ha="left")
intensity_text = ax.text(0.02, 0.71, "", transform=ax.transAxes, va="top", ha="left")
erk_text = ax.text(0.02, 0.66, "", transform=ax.transAxes, va="top", ha="left")

ax.legend(loc="upper right")

active_circles = []
erk_circles = []


def redraw(event_name):
    """Redraw the current state of activation zones, death events and ERK zones."""
    global active_circles, erk_circles

    for circle in active_circles:
        circle.remove()
    active_circles = []

    for circle in erk_circles:
        circle.remove()
    erk_circles = []

    if len(cas_x) > 0:
        cas_scatter.set_offsets(np.column_stack([cas_x, cas_y]))
    else:
        cas_scatter.set_offsets(np.empty((0, 2)))

    if len(death_x) > 0:
        death_scatter.set_offsets(np.column_stack([death_x, death_y]))
    else:
        death_scatter.set_offsets(np.empty((0, 2)))

    if show_rejected_points and len(rej_x) > 0:
        rejected_scatter.set_offsets(np.column_stack([rej_x, rej_y]))
    else:
        rejected_scatter.set_offsets(np.empty((0, 2)))

    if len(active_x) > 0:
        active_scatter.set_offsets(np.column_stack([active_x, active_y]))
    else:
        active_scatter.set_offsets(np.empty((0, 2)))

    if len(erk_x) > 0:
        erk_scatter.set_offsets(np.column_stack([erk_x, erk_y]))
    else:
        erk_scatter.set_offsets(np.empty((0, 2)))

    # Active zones A(V^a_t).
    for x, y, r in zip(active_x, active_y, active_r):
        circle = Circle((x, y), r, fill=False, alpha=0.45, linewidth=1.3)
        ax.add_patch(circle)
        active_circles.append(circle)

    # ERK protected zones A(V^p_t).
    for x, y, r in zip(erk_x, erk_y, erk_r):
        circle = Circle((x, y), r, fill=False, alpha=0.65, linewidth=1.5, linestyle=":")
        ax.add_patch(circle)
        erk_circles.append(circle)

    n_active = len(active_x)
    n_erk = len(erk_x)

    activation_proposal_rate = lambda_a_1 * area_W
    death_proposal_rate = lambda_d * area_W
    active_expiration_rate = n_active * beta_a_T
    erk_expiration_rate = n_erk * beta_d_T
    a0 = (
        activation_proposal_rate
        + death_proposal_rate
        + active_expiration_rate
        + erk_expiration_rate
    )

    time_text.set_text(f"t = {t:.3f}")
    event_text.set_text(f"last event:\n{event_name}")
    count_text.set_text(f"N_active = {n_active} | N_cas = {len(cas_x)}")
    rate_text.set_text(rf"$a_0=\lambda^a_1|W|+\lambda^d|W|+N^a_t\beta^a_T+N^p_t\beta^d_T$ = {a0:.3f}")
    death_text.set_text(f"N_death = {len(death_x)} | active ERK zones = {n_erk}")
    intensity_text.set_text(r"$\lambda^d(x)=\lambda^d 1_{x\in A(V^a_{t-})}1_{x\notin A(V^p_{t-})}$")
    erk_text.set_text(r"$R^d\sim Exp(\beta^d_R),\ \mathrm{ERK\ exp.\ rate}=N^p_t\beta^d_T$")

    fig.canvas.draw_idle()
    plt.pause(display_pause)


# Initial display.
redraw("initialization")


# -----------------------------
# 6. Unified Gillespie / event-driven loop with thinning
# -----------------------------

while t < T_max and (target_deaths is None or len(death_x) < target_deaths):

    n_active = len(active_x)
    n_erk = len(erk_x)

    # Proposal / expiration rates.
    activation_proposal_rate = lambda_a_1 * area_W
    death_proposal_rate = lambda_d * area_W
    active_expiration_rate = n_active * beta_a_T
    erk_expiration_rate = n_erk * beta_d_T

    # Unified Gillespie total rate.
    a0 = (
        activation_proposal_rate
        + death_proposal_rate
        + active_expiration_rate
        + erk_expiration_rate
    )

    if a0 <= 0:
        break

    # Waiting time until the next event among all four possible event types.
    delta_t = rng.exponential(scale=1.0 / a0)
    t += delta_t

    if t > T_max:
        break

    # Choose the event type as in the Gillespie direct method.
    u = rng.uniform(0.0, 1.0)
    p_activation_proposal = activation_proposal_rate / a0
    p_death_proposal = death_proposal_rate / a0
    p_active_expiration = active_expiration_rate / a0

    if u <= p_activation_proposal:
        # Event type 1: candidate activation from dominating PPP(lambda_a_1) on W.
        x = rng.uniform(0.0, Lx)
        y = rng.uniform(0.0, Ly)

        lam_x = local_activation_intensity(x, y)
        accept_prob = lam_x / lambda_a_1
        accept = rng.uniform(0.0, 1.0) <= accept_prob

        if accept:
            cas_x.append(x)
            cas_y.append(y)
            cas_t.append(t)
            cas_origin.append(activation_origin(x, y))

            r = add_active_center(x, y)
            event_name = (
                f"activation accepted from {cas_origin[-1]} "
                f"with p={accept_prob:.3f}, R^a={r:.3f}"
            )
        else:
            rej_x.append(x)
            rej_y.append(y)
            rej_t.append(t)
            rej_type.append("activation")
            event_name = f"activation candidate rejected with p={accept_prob:.3f}"

    elif u <= p_activation_proposal + p_death_proposal:
        # Event type 2: candidate death from PPP(lambda_d) on W.
        # Accept iff it lies inside A(V^a_{t-}) and outside A(V^p_{t-}).
        x = rng.uniform(0.0, Lx)
        y = rng.uniform(0.0, Ly)

        inside_active = is_inside_active_zone(x, y)
        inside_erk = is_inside_erk_protection(x, y)

        if inside_active and not inside_erk:
            death_x.append(x)
            death_y.append(y)
            death_t.append(t)

            r_E = add_erk_protection_zone(x, y)
            event_name = f"death accepted -> ERK protection (R^d={r_E:.3f})"
        else:
            rej_x.append(x)
            rej_y.append(y)
            rej_t.append(t)
            rej_type.append("death")

            if inside_erk:
                event_name = "death candidate rejected: inside ERK protection"
            else:
                event_name = "death candidate rejected: outside active zone"

    elif u <= p_activation_proposal + p_death_proposal + p_active_expiration:
        # Event type 3: expiration of one active center.
        idx = remove_random_active_center()
        if idx is None:
            event_name = "active expiration impossible"
        else:
            event_name = f"active center {idx} expired"

    else:
        # Event type 4: expiration of one ERK protection zone.
        idx = remove_random_erk_zone()
        if idx is None:
            event_name = "ERK expiration impossible"
        else:
            event_name = f"ERK protection zone {idx} expired"

    record_history()
    redraw(event_name)


# -----------------------------
# 7. End
# -----------------------------

if target_deaths is not None and len(death_x) >= target_deaths:
    final_event_name = f"finished: reached {target_deaths} observed deaths"
else:
    final_event_name = "finished: reached T_max"

redraw(final_event_name)

print("Simulation summary")
print(f"  final time: {t:.6f}")
print(f"  accepted activations: {len(cas_x)}")
print(f"  observed deaths: {len(death_x)}")
print(f"  active centers still alive: {len(active_x)}")
print(f"  active ERK zones still alive: {len(erk_x)}")

plt.ioff()
plt.show()
