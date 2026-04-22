import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- CONFIGURACIÓN ---
# Buscamos el archivo en la misma carpeta donde esté este script
ruta_actual = os.path.dirname(__file__)
file_in = os.path.join(ruta_actual, "planets_data.dat")

interval = 20                # Velocidad de la animación (ms)
show_trail = True            # Dibujar la órbita (estela)

# Colores y tamaños ajustados (Sol, Mercurio, Venus, Tierra, Marte, Júpiter)
colores = ['#FFD700', '#A9A9A9', '#FFA500', '#1E90FF', '#FF4500', '#D2B48C']
tamanos = [0.25, 0.05, 0.08, 0.08, 0.06, 0.18] 

# --- LECTURA DE DATOS ---
frames_data = []
try:
    with open(file_in, "r") as f:
        current_frame = []
        for line in f:
            line = line.strip()
            if not line:
                if current_frame:
                    frames_data.append(current_frame)
                    current_frame = []
                continue
            coords = [float(x) for x in line.split(',')]
            current_frame.append(coords)
        if current_frame:
            frames_data.append(current_frame)
except FileNotFoundError:
    print(f"Error: No se encuentra el archivo '{file_in}'")
    print("Asegúrate de ejecutar primero tu programa de simulación.")
    exit()

nplanets = len(frames_data[0])

# --- PREPARACIÓN DE LA FIGURA ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('Simulación Dinámica del Sistema Solar')

limit = 6 # Zoom suficiente para ver hasta Júpiter (5.2 UA)
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal')
ax.set_facecolor('#050505') # Negro espacio
ax.set_title("Sistema Solar: Dinámica de N-Cuerpos", color='white', pad=20)

# --- ESTÉTICA PARA QUE SE VEAN LOS EJES ---
ax.set_facecolor('black')      # Fondo negro
fig.patch.set_facecolor('black') # Fondo de la ventana negro

# Poner los ejes (las líneas de los bordes) en blanco
for spine in ax.spines.values():
    spine.set_color('white')

# Poner los numeritos y las rayitas en blanco
ax.tick_params(colors='white', which='both')

# Poner los nombres de los ejes en blanco
ax.set_xlabel("Distancia (UA)", color='white')
ax.set_ylabel("Distancia (UA)", color='white')
ax.set_title("Simulación del Sistema Solar", color='white')

# Añadir una rejilla muy suave (opcional, queda muy bien)
ax.grid(color='gray', linestyle='--', linewidth=0.2)

  

# Crear los planetas y sus estelas
planet_points = []
planet_trails = []

for i in range(nplanets):
    idx = i % len(colores)
    # El planeta
    c = plt.Circle((0, 0), tamanos[idx], color=colores[idx], zorder=10)
    ax.add_patch(c)
    planet_points.append(c)
    # La estela
    line, = ax.plot([], [], color=colores[idx], linewidth=0.8, alpha=0.5)
    planet_trails.append(line)

# --- FUNCIONES DE ANIMACIÓN ---
def init_anim():
    for i in range(nplanets):
        planet_points[i].center = (0, 0)
        planet_trails[i].set_data([], [])
    return planet_points + planet_trails

def update(j_frame, frames_data, planet_points, planet_trails, show_trail):
    for j_planet, planet_pos in enumerate(frames_data[j_frame]):
        x, y = planet_pos
        planet_points[j_planet].center = (x, y)

        if show_trail:
            # Recuperar historial y añadir punto nuevo
            xs_old, ys_old = planet_trails[j_planet].get_data()
            # Guardamos los últimos 400 puntos para que la estela sea larga pero fluida
            xs_new = np.append(xs_old, x)[-400:]
            ys_new = np.append(ys_old, y)[-400:]
            planet_trails[j_planet].set_data(xs_new, ys_new)

    return planet_points + planet_trails

# --- LANZAMIENTO ---
nframes = len(frames_data)
ani = FuncAnimation(
    fig, update, init_func=init_anim,
    fargs=(frames_data, planet_points, planet_trails, show_trail),
    frames=nframes, blit=True, interval=interval, repeat=True
)

plt.tight_layout()
plt.show()