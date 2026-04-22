import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==============================================================================
# 1. DATOS RESCALADOS PARA CAOS VISUAL (¡Trampa para que sea espectacular!)
# ==============================================================================
M_SOLAR = 1988500.0 
AU_KM = 149.6       

# [Nombre, Masa (10^24kg), Perihelio (10^6km), Excentricidad]
datos = [
    ["Sol",      M_SOLAR, 0.0,    0.0],
    ["Mercurio", 0.330,   46.0,   0.205],
    ["Venus",    4.87,    107.5,  0.007],
    ["Tierra",   5.97,    147.1,  0.017],
    # --- MODIFICACIÓN: Marte es 100 veces más masivo ---
    ["Marte",    64.2,    206.6,  0.094], 
    # --- MODIFICACIÓN: Júpiter es 5 veces más masivo ---
    ["Jupiter",  9490,    740.5,  0.049]  
]

# --- MODIFICACIÓN: Lunas MUCHO más pesadas y cercanas a Júpiter ---
n_lunas = 15
for i in range(n_lunas):
    # Lunas muy cerca de Júpiter (entre 0.1 y 0.8 UA)
    d_relativa = random.uniform(0.1, 0.8) 
    # Masa de las lunas = 500 (¡Enormes, como planetas!)
    datos.append([f"L_{i}", 500.0, 740.5 + (d_relativa * AU_KM), 0.1])

n = len(datos)
m = np.array([p[1] / M_SOLAR for p in datos])
r = np.zeros((n, 2))
v = np.zeros((n, 2))

for i, p in enumerate(datos):
    if i == 0: continue
    r_ua = p[2] / AU_KM
    r[i, 0] = r_ua
    # Velocidad con mucha perturbación aleatoria para caos inmediato
    v[i, 1] = np.sqrt((1 + p[3]) / r_ua) + random.uniform(-0.1, 0.1)

# --- MODIFICACIÓN: Parámetros para que todo pase RÁPIDO ---
h = 0.1        # Paso de tiempo grande = simulación rápida
pasos = 1500   
historico_r = []

# ==============================================================================
# 2. MOTOR FÍSICO OPTIMIZADO
# ==============================================================================
def calcular_aceleracion(pos, masas):
    num_cuerpos = len(masas)
    acc = np.zeros_like(pos)
    # Suavizado pequeño para permitir acercamientos violentos
    eps = 0.02 
    
    for i in range(num_cuerpos):
        for j in range(i + 1, num_cuerpos):
            diff = pos[j] - pos[i]
            dist_sq = np.sum(diff**2) + eps**2
            dist_inv3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
            
            f_grav = diff * dist_inv3
            acc[i] += masas[j] * f_grav
            acc[j] -= masas[i] * f_grav
    return acc

print(f"Calculando desastre visual con {n} cuerpos... Un momento.")
a_actual = calcular_aceleracion(r, m)

for t_step in range(pasos):
    # Guardamos cada paso para que la animación sea una locura
    historico_r.append(r.copy())
    
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual
    a_nueva = calcular_aceleracion(r_nuevo, m)
    v = v + 0.5 * h * (a_actual + a_nueva)
    r, a_actual = r_nuevo, a_nueva

# ==============================================================================
# 3. ANIMACIÓN DE GRAN IMPACTO VISUAL
# ==============================================================================
fig, ax = plt.subplots(figsize=(10,10))
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# --- MODIFICACIÓN: Zoom agresivo (Solo vemos hasta Júpiter) ---
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')

# Colores y tamaños llamativos
colores_planetas = ['yellow', 'grey', 'orange', 'blue', 'red', 'brown']
colores = colores_planetas + ['white'] * n_lunas

# Puntos más grandes para que se vean bien
puntos = [ax.plot([], [], 'o', color=colores[i], ms=8 if i < 6 else 4)[0] for i in range(n)]

def animar(i):
    for j in range(n):
        pos = historico_r[i][j]
        puntos[j].set_data([pos[0]], [pos[1]])
    return puntos

# Animación muy rápida
ani = FuncAnimation(fig, animar, frames=len(historico_r), interval=10, blit=True)

ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('white')
plt.title("EXPERIMENTO: ¡Caos Gravitatorio Total!", color='white', fontsize=16)

plt.show()