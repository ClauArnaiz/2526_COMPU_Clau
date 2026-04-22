import numpy as np

# --- CONSTANTES Y CONFIGURACIÓN ---
G = 1.0  # Gracias al reescalamiento
dt = 0.01 # Paso de tiempo (ajustar según precisión)
pasos = 10000 # Número de iteraciones
M_SOL = 1.0

# Datos: [nombre, masa_relativa, semi_eje_mayor, excentricidad]
# Nota: Puedes añadir más planetas siguiendo este formato
planetas_info = [
    ["Sol", 1.0, 0.0, 0.0],
    ["Mercurio", 1.66e-7, 0.387, 0.2056],
    ["Venus", 2.45e-6, 0.723, 0.0067],
    ["Tierra", 3.00e-6, 1.000, 0.0167],
    ["Marte", 3.23e-7, 1.523, 0.0934],
    ["Jupiter", 9.55e-4, 5.203, 0.0484]
]

num_cuerpos = len(planetas_info)
masas = np.array([p[1] for p in planetas_info])
pos = np.zeros((num_cuerpos, 2))
vel = np.zeros((num_cuerpos, 2))

# --- CONDICIONES INICIALES (PERIHELIO) ---
for i, p in enumerate(planetas_info):
    if i == 0: continue # El Sol empieza en el origen
    a = p[2]
    e = p[3]
    # Posición en el perihelio (x = a*(1-e), y = 0)
    pos[i, 0] = a * (1 - e)
    # Velocidad en el perihelio (v_x = 0, v_y = sqrt(G*M*(1+e)/(a*(1-e))))
    vel[i, 1] = np.sqrt(M_SOL * (1 + e) / (a * (1 - e)))

# --- FUNCIÓN DE ACELERACIÓN ---
def calcular_aceleracion(pos, masas):
    a = np.zeros_like(pos)
    for i in range(num_cuerpos):
        for j in range(num_cuerpos):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_vec)
                a[i] += G * masas[j] * r_vec / r_mag**3
    return a
# Archivo para guardar datos (compatible con el script de la profa)
with open("planets_data.dat", "w") as f:
    
    # Cálculo inicial de aceleración
    acc_actual = calcular_aceleracion(pos, masas)
    
    for t in range(pasos):
        # 1. Guardar posiciones actuales en el archivo
        for i in range(num_cuerpos):
            f.write(f"{pos[i,0]}, {pos[i,1]}\n")
        f.write("\n") # Línea vacía entre instantes de tiempo

        # 2. Actualizar posiciones (Paso 1 de Verlet)
        pos = pos + vel * dt + 0.5 * acc_actual * dt**2
        
        # 3. Calcular nueva aceleración con las nuevas posiciones
        acc_nueva = calcular_aceleracion(pos, masas)
        
        # 4. Actualizar velocidades (Paso 2 de Verlet)
        vel = vel + 0.5 * (acc_actual + acc_nueva) * dt
        
        # 5. La aceleración nueva pasa a ser la actual para el siguiente paso
        acc_actual = acc_nueva