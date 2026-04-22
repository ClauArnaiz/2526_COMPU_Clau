import numpy as np
import matplotlib.pyplot as plt

# Nombres de los planetas (mismo orden que antes, en el script que nos dió los datos usando verlet)
nombres = ["Sol", "Mercurio", "Venus", "Tierra", "Marte"]
n = len(nombres)

# Leer archivo
with open("planets_data.dat", "r") as f:
    bloques = f.read().strip().split("\n\n")

# Inicializar listas
trayectorias = [[] for _ in range(n)]

# Procesar datos
for bloque in bloques:
    lineas = bloque.split("\n")
    for i, linea in enumerate(lineas):
        x, y = map(float, linea.split(","))
        trayectorias[i].append((x, y))

# Dibujar
plt.figure(figsize=(6,6))

for i in range(n):
    datos = np.array(trayectorias[i])
    plt.plot(datos[:,0], datos[:,1], label=nombres[i])

# Mejoras visuales
plt.xlabel("x (UA)")
plt.ylabel("y (UA)")
plt.title("Órbitas planetas rocosos")
plt.legend()
plt.axis("equal")  # MUY IMPORTANTE para que no se deformen las órbitas
plt.grid()

plt.show()