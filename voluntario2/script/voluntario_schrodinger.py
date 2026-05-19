"""
Voluntario 2 - Ecuación de Schrödinger dependiente del tiempo
Oscilador armónico cuántico
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# 1. PARÁMETROS GENERALES
# ============================================================

L = 1.0             # Longitud del sistema
S = 1000            # Número de divisiones espaciales
dx = L / S          # Paso espacial
dt = 0.0001         # Paso temporal
nsteps = 5000       # Número total de iteraciones
guardar_cada = 25   # Guardar un estado cada cierto número de pasos

omega = 200         # Frecuencia angular del oscilador
xc = 0.5            # Centro del potencial armónico

x = np.linspace(0, L, S + 1)
j = np.arange(S + 1)

# ============================================================
# 2. POTENCIAL ARMÓNICO
# ============================================================

def potencial_armonico(x, omega=200, xc=0.5):
    return (omega**2 / 4) * (x - xc)**2

V = potencial_armonico(x, omega, xc)

# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def normalizar(phi):
    norma = np.sqrt(np.sum(np.abs(phi)**2) * dx)
    return phi / norma

def derivada_primera(phi):
    dphi = np.zeros_like(phi, dtype=complex)
    dphi[1:-1] = (phi[2:] - phi[:-2]) / (2 * dx)
    dphi[0] = (phi[1] - phi[0]) / dx
    dphi[-1] = (phi[-1] - phi[-2]) / dx
    return dphi

def derivada_segunda(phi):
    d2phi = np.zeros_like(phi, dtype=complex)
    d2phi[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    return d2phi

def observables(phi, V):
    prob = np.abs(phi)**2

    norma = np.sum(prob) * dx

    x_medio = np.sum(x * prob) * dx
    x2_medio = np.sum(x**2 * prob) * dx

    dphi = derivada_primera(phi)
    d2phi = derivada_segunda(phi)

    p_medio = np.sum(np.conj(phi) * (-1j * dphi)) * dx
    p2_medio = np.sum(np.conj(phi) * (-d2phi)) * dx

    Hphi = -d2phi + V * phi
    energia = np.sum(np.conj(phi) * Hphi) * dx

    # Se usa max(...,0) para evitar errores por redondeos numéricos muy pequeños.
    delta_x = np.sqrt(max(np.real(x2_medio - x_medio**2), 0))
    delta_p = np.sqrt(max(np.real(p2_medio - p_medio**2), 0))

    return {
        "norma": np.real(norma),
        "x": np.real(x_medio),
        "p": np.real(p_medio),
        "energia": np.real(energia),
        "delta_x": delta_x,
        "delta_p": delta_p,
        "incertidumbre": delta_x * delta_p
    }

# ============================================================
# 4. AUTOFUNCIONES DEL OSCILADOR ARMÓNICO
# ============================================================

def hermite(n, z):
    if n == 0:
        return np.ones_like(z)
    if n == 1:
        return 2 * z

    H0 = np.ones_like(z)
    H1 = 2 * z

    for k in range(2, n + 1):
        H2 = 2 * z * H1 - 2 * (k - 1) * H0
        H0, H1 = H1, H2

    return H1

def autofuncion_oscilador(n, x, omega=200, xc=0.5):
    alpha = omega / 2
    z = np.sqrt(alpha) * (x - xc)

    # math.sqrt evita problemas con enteros grandes para n elevado.
    prefactor = (alpha / np.pi)**0.25 / math.sqrt((2**n) * math.factorial(n))
    phi = prefactor * hermite(n, z) * np.exp(-z**2 / 2)

    phi[0] = 0
    phi[-1] = 0

    return normalizar(phi.astype(complex))

def energia_teorica(n, omega=200):
    return omega * (n + 0.5)

# ============================================================
# 5. MÉTODO DE CAYLEY
# ============================================================

def preparar_cayley(V):
    w = dt / dx**2
    V_tilde = dx**2 * V

    A0 = -2 + 2j / w - V_tilde

    alpha = np.zeros(S + 1, dtype=complex)
    alpha[S - 1] = 0

    for jj in range(S - 1, 0, -1):
        gamma = 1 / (A0[jj] + alpha[jj])
        alpha[jj - 1] = -gamma

    return alpha, A0, w

def evolucionar(phi, alpha, A0, w):
    beta = np.zeros(S + 1, dtype=complex)
    beta[S - 1] = 0

    for jj in range(S - 1, 0, -1):
        b = 4j * phi[jj] / w
        gamma = 1 / (A0[jj] + alpha[jj])
        beta[jj - 1] = gamma * (b - beta[jj])

    q = np.zeros(S + 1, dtype=complex)
    q[0] = 0
    q[-1] = 0

    for jj in range(0, S):
        q[jj + 1] = alpha[jj] * q[jj] + beta[jj]

    phi_new = q - phi
    phi_new[0] = 0
    phi_new[-1] = 0

    return phi_new

# ============================================================
# 6. SIMULACIÓN GENERAL
# ============================================================

def simular(phi0, V, nombre="simulacion"):
    alpha, A0, w = preparar_cayley(V)
    phi = phi0.copy()

    phis = []
    tiempos = []

    normas = []
    posiciones = []
    momentos = []
    energias = []
    incertidumbres = []

    for n in range(nsteps + 1):
        if n % guardar_cada == 0:
            phis.append(phi.copy())
            tiempos.append(n * dt)

            obs = observables(phi, V)
            normas.append(obs["norma"])
            posiciones.append(obs["x"])
            momentos.append(obs["p"])
            energias.append(obs["energia"])
            incertidumbres.append(obs["incertidumbre"])

        if n < nsteps:
            phi = evolucionar(phi, alpha, A0, w)

    return {
        "nombre": nombre,
        "phis": np.array(phis),
        "tiempos": np.array(tiempos),
        "normas": np.array(normas),
        "posiciones": np.array(posiciones),
        "momentos": np.array(momentos),
        "energias": np.array(energias),
        "incertidumbres": np.array(incertidumbres)
    }

# ============================================================
# 7. GIFS
# ============================================================

def gif_probabilidad(resultado, V, nombre_gif):
    phis = resultado["phis"]
    tiempos = resultado["tiempos"]

    fig, ax = plt.subplots(figsize=(8, 4))

    prob0 = np.abs(phis[0])**2
    ymax = np.max(prob0) * 1.4

    linea, = ax.plot(x, prob0, label=r"$|\phi(x,t)|^2$")

    if np.max(V) != 0:
        ax.plot(x, V / np.max(V) * np.max(prob0), "--", label="Potencial reescalado")

    ax.set_xlim(0, L)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\phi|^2$")
    ax.legend()

    def actualizar(frame):
        linea.set_ydata(np.abs(phis[frame])**2)
        ax.set_title(f"Probabilidad, t = {tiempos[frame]:.3f}")
        return linea,

    anim = FuncAnimation(fig, actualizar, frames=len(phis), interval=40, blit=True)
    anim.save(nombre_gif, writer=PillowWriter(fps=25))
    plt.close(fig)

    print("GIF guardado en:", os.path.abspath(nombre_gif))

def gif_real_imaginaria(resultado, nombre_gif):
    phis = resultado["phis"]
    tiempos = resultado["tiempos"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ymax = 1.2 * np.max(np.abs(phis[0]))

    linea_re, = ax.plot(x, np.real(phis[0]), label="Re(φ)")
    linea_im, = ax.plot(x, np.imag(phis[0]), label="Im(φ)")

    ax.set_xlim(0, L)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x,t)$")
    ax.legend()

    def actualizar(frame):
        linea_re.set_ydata(np.real(phis[frame]))
        linea_im.set_ydata(np.imag(phis[frame]))
        ax.set_title(f"Parte real e imaginaria, t = {tiempos[frame]:.3f}")
        return linea_re, linea_im

    anim = FuncAnimation(fig, actualizar, frames=len(phis), interval=40, blit=True)
    anim.save(nombre_gif, writer=PillowWriter(fps=25))
    plt.close(fig)

    print("GIF guardado en:", os.path.abspath(nombre_gif))

# ============================================================
# 8. GRÁFICAS DE OBSERVABLES
# ============================================================

def graficar_observables(resultado, titulo="", teoricos=None):
    t = resultado["tiempos"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, resultado["normas"])
    plt.xlabel("t")
    plt.ylabel("Norma")
    plt.title("Conservación de la norma " + titulo)
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(t, resultado["posiciones"], label=r"$\langle x \rangle$")
    if teoricos is not None and "x" in teoricos:
        plt.plot(t, teoricos["x"], "--", label="Teórico / clásico")
    plt.xlabel("t")
    plt.ylabel(r"$\langle x \rangle$")
    plt.title("Valor medio de la posición " + titulo)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(t, resultado["momentos"], label=r"$\langle p \rangle$")
    if teoricos is not None and "p" in teoricos:
        plt.plot(t, teoricos["p"], "--", label="Teórico / clásico")
    plt.xlabel("t")
    plt.ylabel(r"$\langle p \rangle$")
    plt.title("Valor medio del momento " + titulo)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(t, resultado["energias"], label=r"$\langle H \rangle$")
    if teoricos is not None and "E" in teoricos:
        plt.plot(t, teoricos["E"], "--", label="Teórico")
    plt.xlabel("t")
    plt.ylabel(r"$\langle H \rangle$")
    plt.title("Energía media " + titulo)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(t, resultado["incertidumbres"], label=r"$\Delta x \Delta p$")
    plt.axhline(0.5, linestyle="--", label=r"$1/2$")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta x \Delta p$")
    plt.title("Producto de incertidumbres " + titulo)
    plt.legend()
    plt.grid()
    plt.show()

# ============================================================
# 9. COMPARACIÓN CON OSCILADOR CLÁSICO
# ============================================================

def teorico_clasico(resultado):
    t = resultado["tiempos"]

    x0 = resultado["posiciones"][0]
    p0 = resultado["momentos"][0]
    E0 = resultado["energias"][0]

    x_cl = xc + (x0 - xc) * np.cos(omega * t) + (2 * p0 / omega) * np.sin(omega * t)
    p_cl = p0 * np.cos(omega * t) - (omega / 2) * (x0 - xc) * np.sin(omega * t)
    E_cl = np.ones_like(t) * E0

    return {"x": x_cl, "p": p_cl, "E": E_cl}

def teorico_clasico_con_omega(resultado, omega_caso):
    t = resultado["tiempos"]

    x0 = resultado["posiciones"][0]
    p0 = resultado["momentos"][0]
    E0 = resultado["energias"][0]

    x_cl = xc + (x0 - xc) * np.cos(omega_caso * t) + (2 * p0 / omega_caso) * np.sin(omega_caso * t)
    p_cl = p0 * np.cos(omega_caso * t) - (omega_caso / 2) * (x0 - xc) * np.sin(omega_caso * t)
    E_cl = np.ones_like(t) * E0

    return {"x": x_cl, "p": p_cl, "E": E_cl}

# ============================================================
# 10. CASO A: 4 PRIMERAS AUTOFUNCIONES
# ============================================================

def ejecutar_autofunciones():
    resultados_auto = []

    for n in range(4):
        print(f"\n========== Autofunción n = {n} ==========")

        phi0 = autofuncion_oscilador(n, x, omega, xc)
        resultado = simular(phi0, V, nombre=f"autofuncion_n{n}")
        resultados_auto.append(resultado)

        E_teo = energia_teorica(n, omega)
        teoricos = {
            "x": np.ones_like(resultado["tiempos"]) * xc,
            "p": np.zeros_like(resultado["tiempos"]),
            "E": np.ones_like(resultado["tiempos"]) * E_teo
        }

        print("Energía teórica:", E_teo)
        print("Energía numérica inicial:", resultado["energias"][0])
        print("Norma inicial:", resultado["normas"][0])
        print("Norma final:", resultado["normas"][-1])

        graficar_observables(resultado, titulo=f"(autofunción n={n})", teoricos=teoricos)

    gif_real_imaginaria(resultados_auto[0], "auto_n0_real_imag.gif")
    gif_probabilidad(resultados_auto[0], V, "auto_n0_probabilidad.gif")

    return resultados_auto

# ============================================================
# 11. CASO B: FUNCIÓN GAUSSIANA
# ============================================================

def gaussiana(x0=0.3, sigma=1/16):
    phi = np.exp(-(x - x0)**2 / (2 * sigma**2)).astype(complex)
    phi[0] = 0
    phi[-1] = 0
    return normalizar(phi)

def ejecutar_gaussiana():
    phi_gauss = gaussiana(x0=0.3, sigma=1/16)

    resultado_gauss = simular(phi_gauss, V, nombre="gaussiana_x03_sigma116")
    teorico_gauss = teorico_clasico(resultado_gauss)

    print("\n========== Gaussiana x0=0.3, sigma=1/16 ==========")
    print("Norma inicial:", resultado_gauss["normas"][0])
    print("Norma final:", resultado_gauss["normas"][-1])
    print("Energía inicial:", resultado_gauss["energias"][0])

    graficar_observables(resultado_gauss, titulo="(gaussiana)", teoricos=teorico_gauss)

    gif_real_imaginaria(resultado_gauss, "gaussiana_real_imag.gif")
    gif_probabilidad(resultado_gauss, V, "gaussiana_probabilidad.gif")

    return resultado_gauss

# ============================================================
# 12. CASOS EXTRA
# ============================================================

def ejecutar_casos_extra():
    casos_extra = [
        {"x0": 0.5, "sigma": 1/16, "omega": 200, "nombre": "gaussiana_x05_sigma116"},
        {"x0": 0.5, "sigma": 1/10, "omega": 200, "nombre": "gaussiana_x05_sigma110"},
        {"x0": 0.3, "sigma": 1/16, "omega": 100, "nombre": "gaussiana_x03_sigma116_omega100"},
        {"x0": 0.3, "sigma": 1/16, "omega": 300, "nombre": "gaussiana_x03_sigma116_omega300"}
    ]

    resultados_extra = []

    for caso in casos_extra:
        print(f"\n========== Caso extra: {caso['nombre']} ==========")

        omega_caso = caso["omega"]
        V_caso = potencial_armonico(x, omega_caso, xc)

        phi0 = gaussiana(x0=caso["x0"], sigma=caso["sigma"])
        resultado = simular(phi0, V_caso, nombre=caso["nombre"])
        teorico = teorico_clasico_con_omega(resultado, omega_caso)

        resultados_extra.append(resultado)

        print("Norma inicial:", resultado["normas"][0])
        print("Norma final:", resultado["normas"][-1])
        print("Energía inicial:", resultado["energias"][0])

        graficar_observables(resultado, titulo=f"({caso['nombre']})", teoricos=teorico)

    return resultados_extra

# ============================================================
# 13. PRINCIPIO DE CORRESPONDENCIA: n GRANDE
# ============================================================

def principio_correspondencia():
    n_grande = 20

    phi_n20 = autofuncion_oscilador(n_grande, x, omega=200, xc=0.5)
    prob_n20 = np.abs(phi_n20)**2

    E20 = energia_teorica(n_grande, omega=200)

    x_min = xc - 2 * np.sqrt(E20) / 200
    x_max = xc + 2 * np.sqrt(E20) / 200

    prob_cl = np.zeros_like(x)

    eps = 1e-6
    zona = (x > x_min + eps) & (x < x_max - eps)

    prob_cl[zona] = 1 / np.sqrt((x[zona] - x_min) * (x_max - x[zona]))

    prob_cl = prob_cl / (np.sum(prob_cl) * dx)

    plt.figure(figsize=(8, 4))
    plt.plot(x, prob_n20, label="Probabilidad cuántica n=20")
    plt.plot(x, prob_cl, "--", label="Probabilidad clásica reescalada")
    plt.xlabel("x")
    plt.ylabel("Probabilidad")
    plt.title("Principio de correspondencia para n = 20")
    plt.legend()
    plt.grid()
    plt.show()

# ============================================================
# EJECUCIÓN DEL PROGRAMA
# ============================================================

if __name__ == "__main__":
    resultados_auto = ejecutar_autofunciones()
    resultado_gauss = ejecutar_gaussiana()
    resultados_extra = ejecutar_casos_extra()
    principio_correspondencia()
