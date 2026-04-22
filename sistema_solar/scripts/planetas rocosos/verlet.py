import numpy as np

# 1. DATOS INICIALES (El "Vector" de datos reales)
# Unidades: G=1, M_sol=1, Distancia=UA
# Datos para planetas rocosos (Unidades reescaladas)
nombres = ["Sol", "Mercurio", "Venus", "Tierra", "Marte"]
masas   = np.array([1.0, 1.66e-7, 2.45e-6, 3.00e-6, 3.23e-7])
dist    = np.array([0.0, 0.387, 0.723, 1.000, 1.523])
excent  = np.array([0.0, 0.2056, 0.0067, 0.0167, 0.0934])

n = len(masas)
pos = np.zeros((n, 2)) #posición (x,y) para cada planeta
vel = np.zeros((n, 2)) #velocidad (vx, vy) para cada planeta

# Inicialización (Perihelio)
for i in range(n):
    if i == 0: continue # Sol en el origen, es decir, si va por el Sol, lo evitamos

    # Posición inicial en el eje X
    pos[i, 0] = dist[i] * (1 - excent[i])

    # Velocidad inicial en el eje Y (Fórmula de la teoría) ---- Esto viene de conservación de energía + momento angular ----- En el perihelio la velocidad es máxima y perpendicular al radio.
    vel[i, 1] = np.sqrt((1 + excent[i]) / pos[i, 0])


# 2. FUNCIÓN ACELERACIÓN (Interacción de todos con todos, ya que los planetas ejercen gravedad uno sobre otro, aunque el SOl domina, los demás también se afectan entre si)
def calcular_aceleracion(p, m):
    a = np.zeros_like(p)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = p[j] - p[i]
                norma = np.linalg.norm(r_ij)
                a[i] += m[j] * r_ij / norma**3 # Ley de gravitación universal en forma verctorial: a = G * m_j * r_ij / |r_ij|^3, con G=1 en nuestras unidades
    return a

# 3. BUCLE PRINCIPAL (Algoritmo de Verlet)
dt = 0.001  # Paso más pequeño para mayor precisión con rocosos
pasos = 20000 # Simular por 20 unidades de tiempo (20 años en este caso)

acc = calcular_aceleracion(pos, masas)

with open("planets_data.dat", "w") as f: # Guardar datos para visualización posterior y poder usarlos para hacer las gŕaficas

    for t in range(pasos):
        # 1. Escribir posiciones actuales
        for i in range(n):
            f.write(f"{pos[i,0]}, {pos[i,1]}\n")
        f.write("\n")

        #Cada planeta una línea
        # Línea en blanco separa cada instante temporal

        # 2. ALGORITMO DE VERLET
        # Paso A: Nueva posición r(t+dt)
        pos = pos + vel * dt + 0.5 * acc * dt**2
        
        # Paso B: Nueva aceleración a(t+dt) con la posición nueva
        acc_nueva = calcular_aceleracion(pos, masas)
        
        # Paso C: Nueva velocidad v(t+dt)
        vel = vel + 0.5 * (acc + acc_nueva) * dt
        
        # Actualizar para el siguiente paso
        acc = acc_nueva