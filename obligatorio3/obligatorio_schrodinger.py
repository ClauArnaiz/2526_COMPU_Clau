import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
import os

# ============================================================
# 1. PARÁMETROS
# ============================================================

N = 1000
L = 1.0
h = L / N

nciclos = 50
lam = 0.3

assert 1 <= nciclos <= N / 4, "nciclos debe estar entre 1 y N/4"

k_tilde = 2 * np.pi * nciclos / N
s_tilde = 1 / (4 * k_tilde**2)
s = s_tilde * h**2

nsteps = 4000
guardar_cada = 20

x = np.linspace(0, L, N + 1)
j = np.arange(N + 1)

# ============================================================
# 2. FUNCIÓN DE ONDA INICIAL
# ============================================================

x0 = L / 4
sigma = L / 16

phi = np.exp(1j * k_tilde * j) * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Condiciones de contorno: paredes infinitas
phi[0] = 0
phi[-1] = 0

# Normalización inicial
norma_inicial = np.sqrt(np.sum(np.abs(phi)**2) * h)
phi = phi / norma_inicial

# ============================================================
# 3. POTENCIAL CUADRADO
# ============================================================

V_tilde = np.zeros(N + 1)

j1 = int(2 * N / 5)
j2 = int(3 * N / 5)

# Potencial cuadrado centrado en N/2, de anchura N/5
# Se incluye j2 porque el intervalo teórico es [2N/5, 3N/5]
V_tilde[j1:j2 + 1] = lam * k_tilde**2

# ============================================================
# 4. COEFICIENTES DEL SISTEMA TRIDIAGONAL
# ============================================================

A_minus = 1.0
A_plus = 1.0

A0 = -2 + 2j / s_tilde - V_tilde

alpha = np.zeros(N + 1, dtype=complex)
alpha[N - 1] = 0

# Cálculo de alpha, que no depende del tiempo
for jj in range(N - 1, 0, -1):
    gamma = 1 / (A0[jj] + A_plus * alpha[jj])
    alpha[jj - 1] = -A_minus * gamma

# ============================================================
# 5. FUNCIÓN DE EVOLUCIÓN TEMPORAL
# ============================================================

def evolucionar(phi):
    beta = np.zeros(N + 1, dtype=complex)
    beta[N - 1] = 0

    # Cálculo de beta
    for jj in range(N - 1, 0, -1):
        b = 4j * phi[jj] / s_tilde
        gamma = 1 / (A0[jj] + A_plus * alpha[jj])
        beta[jj - 1] = gamma * (b - A_plus * beta[jj])

    # Cálculo de chi usando la recurrencia hacia delante
    chi = np.zeros(N + 1, dtype=complex)
    chi[0] = 0
    chi[-1] = 0

    for jj in range(0, N):
        chi[jj + 1] = alpha[jj] * chi[jj] + beta[jj]

    # Evolución temporal mediante el método de Cayley
    phi_new = chi - phi

    # Condiciones de contorno
    phi_new[0] = 0
    phi_new[-1] = 0

    return phi_new

# ============================================================
# 6. SIMULACIÓN
# ============================================================

phis = []
normas = []
tiempos = []

for n in range(nsteps + 1):
    if n % guardar_cada == 0:
        phis.append(phi.copy())
        normas.append(np.sum(np.abs(phi)**2) * h)
        tiempos.append(n * s)

    if n < nsteps:
        phi = evolucionar(phi)

phis = np.array(phis)
normas = np.array(normas)
tiempos = np.array(tiempos)

# Estado final real de la simulación
phi_final = phi.copy()

# ============================================================
# 7. GRÁFICA DE LA NORMA
# ============================================================

plt.figure()
plt.plot(tiempos, normas)
plt.xlabel("t")
plt.ylabel("Norma")
plt.title("Conservación de la norma")
plt.ticklabel_format(useOffset=False)
plt.ylim(0.999999999999, 1.000000000001)
plt.grid()
plt.show()

plt.figure()
plt.plot(tiempos, normas - 1)
plt.xlabel("t")
plt.ylabel("Norma - 1")
plt.title("Error en la conservación de la norma")
plt.grid()
plt.show()

# ============================================================
# 8. ANIMACIÓN DE |PHI|²
# ============================================================

fig, ax = plt.subplots()

prob0 = np.abs(phis[0])**2
prob_max = np.max(prob0)

linea, = ax.plot(x, prob0, label=r"$|\Phi(x,t)|^2$")

if np.max(V_tilde) > 0:
    ax.plot(
        x,
        V_tilde / np.max(V_tilde) * prob_max,
        "--",
        label="Potencial reescalado"
    )

ax.set_xlim(0, L)
ax.set_ylim(0, prob_max * 1.4)
ax.set_xlabel("x")
ax.set_ylabel(r"$|\Phi|^2$")
ax.legend()

def actualizar(frame):
    prob = np.abs(phis[frame])**2
    linea.set_ydata(prob)
    ax.set_title(f"Evolución de |Phi|², t = {tiempos[frame]:.5f}")
    return linea,

anim = FuncAnimation(
    fig,
    actualizar,
    frames=len(phis),
    interval=40,
    blit=True
)

nombre_gif = "schrodinger_obligatorio.gif"

anim.save(nombre_gif, writer=PillowWriter(fps=25))

plt.close(fig)

print("GIF guardado en:")
print(os.path.abspath(nombre_gif))

display(Image(filename=nombre_gif))

# ============================================================
# 9. COEFICIENTES DE REFLEXIÓN, BARRERA Y TRANSMISIÓN
# ============================================================

prob_final = np.abs(phi_final)**2

R = np.sum(prob_final[:j1] * h)
B = np.sum(prob_final[j1:j2 + 1] * h)
T = np.sum(prob_final[j2 + 1:] * h)

print("Coeficiente de reflexión aproximado R =", R)
print("Probabilidad dentro de la barrera B =", B)
print("Coeficiente de transmisión aproximado T =", T)
print("R + B + T =", R + B + T)
print("Norma final =", np.sum(prob_final) * h)

# Nota:
# R y T son aproximados. Tienen sentido físico cuando el paquete de ondas
# ya ha interactuado con la barrera, pero antes de reflejarse en la pared derecha.
