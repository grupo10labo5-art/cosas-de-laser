"""
Toy simulation: output power vs pump current for a diode-pumped Nd:YAG laser
in a plane-plane (flat-flat) cavity, healthy rod vs cracked rod.

Physics included (simple, semi-quantitative):
  * Pump power is linear in current: P_pump = k * (I - I_on).
  * Four-level laser: threshold ~ (T + L_rt), slope ~ T / (T + L_rt).
  * A flat-flat cavity is only stabilised by the thermal lens of the rod
    (f_th = K / P_pump). With the rod in the middle, g = 1 - d / f_th:
      - low pump  -> f_th huge, g -> 1: marginally stable, big diffraction loss
      - high pump -> f_th short, g -> -1: cavity goes unstable, laser dies
  * Cracks add:
      1) fixed scattering loss at the crack surfaces
      2) a stronger, aberrated thermal lens (worse heat flow across cracks)
      3) aberration / depolarisation loss growing ~ P_pump^2
      4) some output jitter (hot spots, mode hopping across the fracture)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- constants / parameters ----------------
lam_p, lam_l = 808e-9, 1064e-9          # pump and laser wavelengths [m]
h, c = 6.626e-34, 3e8
sigma = 2.8e-23                          # stim. emission cross section [m^2]
tau = 230e-6                             # upper-level lifetime [s]
w0 = 300e-6                              # mode radius in the rod [m]
A_mode = np.pi * w0**2
eta_abs = 0.90                           # fraction of pump absorbed
eta_ovl = 0.85                           # pump/mode overlap
T_oc = 0.10                              # output coupler transmission
d = 0.15                                 # rod-to-mirror distance [m] (rod centered)

k_diode, I_on = 1.1, 0.8                 # P_pump = k (I - I_on)  [W/A, A]
I = np.linspace(0, 25, 600)              # pump current [A]
P_pump = np.clip(k_diode * (I - I_on), 0, None)

# gain coefficient: round-trip gain = P_pump / C
C = h * c / lam_p * A_mode / (2 * sigma * tau * eta_abs)


def output_power(P, L_base, K_lens, beta_ab, jitter=0.0, seed=0):
    """Output power [W] for pump power array P."""
    Pm = np.maximum(P, 1e-6)
    f_th = K_lens / Pm                           # thermal focal length [m]
    g = 1 - d / f_th                             # equivalent g-parameter
    stable = np.abs(g) < 1

    # diffraction loss blows up near both stability edges (|g| -> 1)
    L_diff = np.where(stable, 0.003 / np.maximum(1 - g**2, 1e-9), np.inf)
    L_ab = beta_ab * P**2                        # aberration/depolarisation
    L_rt = L_base + L_diff + L_ab                # total internal round-trip loss

    P_th = C * (T_oc + L_rt)
    eta_slope = eta_abs * eta_ovl * (lam_p / lam_l) * T_oc / (T_oc + L_rt)
    Pout = np.where(P > P_th, eta_slope * (P - P_th), 0.0)

    if jitter:
        rng = np.random.default_rng(seed)
        Pout = Pout * (1 + jitter * rng.standard_normal(P.size))
    return np.clip(Pout, 0, None), P_th


# healthy rod
P_ok, Pth_ok = output_power(P_pump, L_base=0.02, K_lens=4.0, beta_ab=0.0)
# cracked rod
P_cr, Pth_cr = output_power(P_pump, L_base=0.02 + 0.06, K_lens=1.6,
                            beta_ab=3e-4, jitter=0.06, seed=1)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(I, P_ok, lw=2, label="Healthy Nd:YAG rod")
ax.plot(I, P_cr, lw=1.5, label="Cracked Nd:YAG rod")

for P_out, name in [(P_ok, "healthy"), (P_cr, "cracked")]:
    lasing = I[P_out > 0]
    if lasing.size:
        print(f"{name:8s}: threshold ~ {lasing[0]:5.2f} A, "
              f"lasing stops ~ {lasing[-1]:5.2f} A, "
              f"max P_out = {P_out.max():.2f} W")

ax.set_xlabel("Pump diode current  I  [A]")
ax.set_ylabel("Laser output power  [W]")
ax.set_title("Nd:YAG, plane-plane cavity: healthy vs cracked rod")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("ndyag_cracked_cavity.png", dpi=150)
plt.show()
