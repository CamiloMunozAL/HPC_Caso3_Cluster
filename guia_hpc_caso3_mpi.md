# Guía paso a paso — Caso 3 HPC: Clúster en Google Cloud y Multiplicación de Matrices con MPI

## 1. Objetivo del trabajo

El objetivo del caso de estudio es implementar una solución paralela usando **MPI** sobre el caso de estudio de **multiplicación de matrices**, ejecutarla sobre un **clúster computacional** y realizar pruebas de desempeño.

La actividad requiere:

- Implementar una versión paralelizada con MPI.
- Ejecutar las pruebas sobre un clúster computacional.
- Usar mínimo 3 nodos de cómputo.
- Comparar los resultados contra una versión secuencial.
- Presentar resultados en un documento técnico.

En nuestro caso, se decidió usar Google Cloud con la siguiente arquitectura:

```text
head  -> nodo principal
wn1   -> nodo trabajador 1
wn2   -> nodo trabajador 2
wn3   -> nodo trabajador 3
```

Aunque `head` cumple el rol de nodo principal, también puede participar en el cómputo durante las pruebas MPI.

---

## 2. Arquitectura del clúster

Se crearon 4 instancias en Google Cloud tipo `e2-small`, todas en la misma zona:

```text
head
wn1
wn2
wn3
```

Ejemplo de IPs internas usadas:

```text
head  10.128.0.2
wn1   10.128.0.3
wn2   10.128.0.4
wn3   10.128.0.5
```

La IP pública solo se usa para conectarse desde el portátil al nodo `head`. La comunicación interna entre nodos debe hacerse con las **IP privadas internas**.

---

## 3. Configuración de nombres de nodos

En cada máquina se configuró el hostname correspondiente.

En `head`:

```bash
sudo hostnamectl set-hostname head
```

En `wn1`:

```bash
sudo hostnamectl set-hostname wn1
```

En `wn2`:

```bash
sudo hostnamectl set-hostname wn2
```

En `wn3`:

```bash
sudo hostnamectl set-hostname wn3
```

Después de cambiar el hostname se recomienda reiniciar:

```bash
sudo reboot
```

Para verificar:

```bash
hostname
```

---

## 4. Configuración de `/etc/hosts`

En cada nodo se debe editar:

```bash
sudo nano /etc/hosts
```

Agregar las IPs internas y nombres:

```text
10.128.0.2 head
10.128.0.3 wn1
10.128.0.4 wn2
10.128.0.5 wn3
```

Esto permite conectarse usando nombres:

```bash
ssh hpcuser@wn1
```

Si `ping` no existe:

```bash
sudo apt update
sudo apt install iputils-ping -y
```

Verificación desde `head`:

```bash
ping wn1
ping wn2
ping wn3
```

---

## 5. Usuario de trabajo `hpcuser`

Se creó un usuario común:

```bash
sudo adduser hpcuser
sudo usermod -aG sudo hpcuser
```

Este usuario debe existir en:

```text
head
wn1
wn2
wn3
```

El propósito de usar `hpcuser` es tener un usuario común para compilar, ejecutar MPI, sincronizar archivos y permitir acceso al equipo de trabajo.

---

## 6. SSH sin contraseña

MPI necesita que el nodo principal pueda lanzar procesos en los nodos trabajadores sin pedir contraseña.

La idea clave es:

```text
La clave pública de hpcuser en head debe estar en:

wn1:/home/hpcuser/.ssh/authorized_keys
wn2:/home/hpcuser/.ssh/authorized_keys
wn3:/home/hpcuser/.ssh/authorized_keys
```

### 6.1 Crear llave SSH en `head`

Entrar como `hpcuser`:

```bash
sudo su - hpcuser
```

Generar llave:

```bash
ssh-keygen
```

Aceptar todo con Enter.

Ver la clave pública:

```bash
cat ~/.ssh/id_rsa.pub
```

Copiar esa línea completa.

### 6.2 Agregar la clave pública en cada worker

En cada nodo trabajador:

```bash
sudo su - hpcuser
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys
```

Pegar la clave pública de `head`.

Configurar permisos:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 6.3 Probar SSH desde `head`

```bash
ssh hpcuser@wn1
exit
ssh hpcuser@wn2
exit
ssh hpcuser@wn3
exit
```

Si no pide contraseña, la configuración está correcta.

---

## 7. Instalación de MPI

Se recomienda OpenMPI por su instalación sencilla en Ubuntu.

En cada nodo:

```bash
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev build-essential -y
```

Verificar:

```bash
mpirun --version
mpicc --version
```

---

## 8. Prueba inicial de MPI

Antes de programar matrices, validar que MPI ejecuta procesos en varios nodos.

Desde `head`:

```bash
mpirun -np 4 --host head:1,wn1:1,wn2:1,wn3:1 hostname
```

Salida esperada:

```text
head
wn1
wn2
wn3
```

Con 8 procesos:

```bash
mpirun -np 8 --host head:2,wn1:2,wn2:2,wn3:2 hostname
```

Esto ejecuta 2 procesos por nodo. Importante: **8 procesos no son 8 nodos**.

---

## 9. Conceptos fundamentales de MPI

### 9.1 ¿Qué es MPI?

MPI significa **Message Passing Interface**. Es una interfaz de paso de mensajes que permite que varios procesos se comuniquen entre sí en una arquitectura de memoria distribuida.

En memoria distribuida:

```text
Cada proceso tiene su propia memoria.
Los procesos no comparten variables directamente.
La comunicación ocurre mediante mensajes.
```

### 9.2 Diferencia entre hilos y MPI

| Versión con hilos | Versión MPI |
|---|---|
| Usa `pthread_create` | Usa `mpirun` |
| Los hilos comparten memoria | Cada proceso tiene su propia memoria |
| Se ejecuta en una máquina | Puede ejecutarse en varias máquinas |
| Se reparte trabajo con `row_start` y `row_end` | Se reparte trabajo con `MPI_Scatter` |
| Se espera con `pthread_join` | Se sincroniza con `MPI_Barrier` |
| Se mide con `clock_gettime` | Se mide con `MPI_Wtime` |

### 9.3 `MPI_Init`

```c
MPI_Init(&argc, &argv);
```

Inicializa el entorno MPI. Todo programa MPI debe llamarlo antes de usar cualquier otra función MPI.

### 9.4 `MPI_Finalize`

```c
MPI_Finalize();
```

Finaliza el entorno MPI. Después de esta función no deben llamarse más funciones MPI.

### 9.5 `MPI_Comm_rank`

```c
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

Obtiene el identificador del proceso actual.

Ejemplo con 4 procesos:

```text
rank 0
rank 1
rank 2
rank 3
```

Normalmente `rank 0` se usa como proceso principal.

### 9.6 `MPI_Comm_size`

```c
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

Obtiene la cantidad total de procesos MPI.

Si se ejecuta:

```bash
mpirun -np 4 ./programa
```

entonces:

```text
size = 4
```

### 9.7 `MPI_COMM_WORLD`

Es el comunicador por defecto que contiene todos los procesos lanzados con `mpirun`.

---

## 10. Comunicación colectiva usada en la multiplicación de matrices

### 10.1 `MPI_Bcast`

```c
MPI_Bcast(B, full_elems, MPI_INT32_T, 0, MPI_COMM_WORLD);
```

Envía la matriz `B` completa desde `rank 0` a todos los procesos.

Se usa porque todos los procesos necesitan la matriz `B` completa para calcular sus filas de `C`.

### 10.2 `MPI_Scatter`

```c
MPI_Scatter(
    A,
    local_elems,
    MPI_INT32_T,
    local_A,
    local_elems,
    MPI_INT32_T,
    0,
    MPI_COMM_WORLD
);
```

Divide la matriz `A` en bloques de filas y entrega un bloque a cada proceso.

Ejemplo con `N = 1000` y `4 procesos`:

```text
rank 0 recibe filas 0-249
rank 1 recibe filas 250-499
rank 2 recibe filas 500-749
rank 3 recibe filas 750-999
```

### 10.3 `MPI_Gather`

```c
MPI_Gather(
    local_C,
    local_elems,
    MPI_INT32_T,
    C,
    local_elems,
    MPI_INT32_T,
    0,
    MPI_COMM_WORLD
);
```

Recoge los bloques calculados por cada proceso y los reúne en la matriz `C` completa dentro de `rank 0`.

### 10.4 `MPI_Barrier`

```c
MPI_Barrier(MPI_COMM_WORLD);
```

Sincroniza todos los procesos. Se usa antes y después de medir tiempo para que todos los procesos inicien y terminen la medición de manera coordinada.

### 10.5 `MPI_Wtime`

```c
double t0 = MPI_Wtime();
double t1 = MPI_Wtime();
```

Mide el tiempo de pared en programas MPI. Es preferible a `clock_gettime` cuando hay múltiples procesos distribuidos.

---

## 11. Estrategia de paralelización de matrices

La multiplicación clásica es:

```text
C = A x B
```

Secuencialmente:

```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

En MPI:

```text
1. rank 0 crea A, B y C completas.
2. rank 0 reparte A por filas usando MPI_Scatter.
3. rank 0 envía B completa a todos usando MPI_Bcast.
4. Cada proceso calcula su bloque local de C.
5. rank 0 reúne C completa usando MPI_Gather.
6. rank 0 imprime tiempo y checksum.
```

Representación:

```text
A completa                 B completa
    |                           |
MPI_Scatter                 MPI_Bcast
    |                           |
local_A en cada proceso     B en todos
    |
matmul_block()
    |
local_C
    |
MPI_Gather
    |
C completa en rank 0
```

---

## 12. Código ejemplo mínimo MPI

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hola desde rank %d de %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

Compilar:

```bash
mpicc hello_mpi.c -o hello_mpi
```

Ejecutar:

```bash
mpirun -np 4 --host head:1,wn1:1,wn2:1,wn3:1 ./hello_mpi
```

---

## 13. Organización del repositorio

En `head` se creó:

```bash
~/HPC_Caso3_Cluster
```

Estructura recomendada:

```text
HPC_Caso3_Cluster/
├── matmul_seq.c
├── matmul_seq
├── matmul_mpi.c
├── matmul_mpi
├── run_seq.sh
├── run_mpi.sh
├── sync_workers.sh
├── output/
│   ├── output_seq.txt
│   └── output_mpi.txt
└── README.md
```

---

## 14. ¿Por qué copiar el binario a cada nodo?

Los nodos no comparten automáticamente el mismo sistema de archivos.

Si el binario existe solo en `head`, al ejecutar:

```bash
mpirun -np 4 --host head,wn1,wn2,wn3 ./matmul_mpi
```

los workers pueden fallar porque `wn1`, `wn2` y `wn3` no tienen el archivo.

Por eso se recomienda que el binario exista en la misma ruta en todos los nodos:

```text
/home/hpcuser/HPC_Caso3_Cluster/matmul_mpi
```

---

## 15. Script para sincronizar workers

Archivo: `sync_workers.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="$HOME/HPC_Caso3_Cluster"
WORKERS=(wn1 wn2 wn3)

if [ ! -d "$DIR" ]; then
  echo "No existe el directorio: $DIR" >&2
  exit 1
fi

for node in "${WORKERS[@]}"; do
  echo "Sincronizando con $node..."
  ssh "hpcuser@$node" "mkdir -p '$DIR'"
  rsync -av --delete "$DIR/" "hpcuser@$node:$DIR/"
done

echo "Sincronización completada."
```

Dar permisos:

```bash
chmod +x sync_workers.sh
```

Ejecutar:

```bash
./sync_workers.sh
```

---

## 16. Compilación de la versión secuencial

```bash
gcc -O3 -Wall -Wextra matmul_seq.c -o matmul_seq
```

Ejemplo:

```bash
./matmul_seq 400 4 123456789
```

---

## 17. Compilación de la versión MPI

```bash
mpicc -O3 -Wall -Wextra matmul_mpi.c -o matmul_mpi
```

Luego sincronizar:

```bash
./sync_workers.sh
```

---

## 18. Ejecuciones MPI recomendadas

1 proceso:

```bash
mpirun -np 1 --host head:1 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

2 procesos:

```bash
mpirun -np 2 --host head:1,wn1:1 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

4 procesos:

```bash
mpirun -np 4 --host head:1,wn1:1,wn2:1,wn3:1 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

8 procesos:

```bash
mpirun -np 8 --host head:2,wn1:2,wn2:2,wn3:2 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

---

## 19. Script de benchmark secuencial

Archivo: `run_seq.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/matmul_seq.c"
BIN="$DIR/matmul_seq"
OUT_DIR="$DIR/output"
OUT="$OUT_DIR/output_seq.txt"
SEED=123456789
ROUNDS=4
SIZES=(400 600 800 1000 1200 1600)

mkdir -p "$OUT_DIR"

gcc -O3 -Wall -Wextra "$SRC" -o "$BIN"
echo "N version trial wall_s user_s kernel_s cpu_total_s checksum seed" > "$OUT"

for round in $(seq 1 "$ROUNDS"); do
  for N in "${SIZES[@]}"; do
    "$BIN" "$N" 1 "$SEED" | awk 'NR > 1' >> "$OUT"
  done
done

echo "Resultados guardados en: $OUT"
```

---

## 20. Script de benchmark MPI

Archivo: `run_mpi.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/matmul_mpi.c"
BIN="$DIR/matmul_mpi"
OUT_DIR="$DIR/output"
OUT="$OUT_DIR/output_mpi.txt"
SEED=123456789
ROUNDS=4
SIZES=(400 600 800 1000 1200 1600)
PROCS=(1 2 4 8)
WORKERS=(wn1 wn2 wn3)

mkdir -p "$OUT_DIR"

mpicc -O3 -Wall -Wextra "$SRC" -o "$BIN"

for node in "${WORKERS[@]}"; do
  scp "$BIN" "hpcuser@$node:$BIN"
done

echo "N processes trial wall_s checksum seed" > "$OUT"

for round in $(seq 1 "$ROUNDS"); do
  for N in "${SIZES[@]}"; do
    for P in "${PROCS[@]}"; do
      if [ "$P" -eq 1 ]; then
        HOSTS="head:1"
      elif [ "$P" -eq 2 ]; then
        HOSTS="head:1,wn1:1"
      elif [ "$P" -eq 4 ]; then
        HOSTS="head:1,wn1:1,wn2:1,wn3:1"
      elif [ "$P" -eq 8 ]; then
        HOSTS="head:2,wn1:2,wn2:2,wn3:2"
      else
        echo "Número de procesos no soportado: $P" >&2
        exit 1
      fi

      mpirun -np "$P" --host "$HOSTS" "$BIN" "$N" 1 "$SEED" | awk 'NR > 1' >> "$OUT"
    done
  done
done

echo "Resultados guardados en: $OUT"
```

---

## 21. Diseño experimental recomendado

```text
N = 400, 600, 800, 1000, 1200, 1600
Procesos MPI = 1, 2, 4, 8
Repeticiones = 4
Seed = 123456789
```

---

## 22. Cálculo de speedup

```text
Speedup = tiempo_secuencial / tiempo_MPI
```

Ejemplo:

```text
Tiempo secuencial = 10.0 s
Tiempo MPI 4 procesos = 3.2 s
Speedup = 10.0 / 3.2 = 3.125
```

---

## 23. Cálculo de eficiencia

```text
Eficiencia = Speedup / número de procesos
```

Ejemplo:

```text
Speedup = 3.125
Procesos = 4
Eficiencia = 3.125 / 4 = 0.78125 = 78.12%
```

---

## 24. Análisis esperado

En matrices pequeñas, MPI puede ser más lento que la versión secuencial por el costo de comunicación:

```text
MPI_Scatter
MPI_Bcast
MPI_Gather
sincronización
lanzamiento de procesos
```

A medida que `N` aumenta, el trabajo computacional crece y puede amortizar mejor el costo de comunicación.

---

## 25. Uso de Git dentro del clúster

Se puede usar Git desde `head`:

```bash
cd ~/HPC_Caso3_Cluster
git status
git add .
git commit -m "Implementacion MPI y scripts de benchmark"
git push origin main
```

Los workers no necesitan manejar Git. El flujo correcto es:

```text
head:
  gestiona Git
  compila
  sincroniza workers

wn1, wn2, wn3:
  solo ejecutan el binario sincronizado
```

---

## 26. Estructura sugerida del informe

```text
1. Introducción
2. Descripción del clúster
3. Fundamentos de MPI
4. Versión secuencial base
5. Diseño de la versión paralela con MPI
6. Configuración experimental
7. Resultados
8. Speedup y eficiencia
9. Análisis de desempeño
10. Conclusiones
11. Enlace al repositorio
```

---

## 27. Texto base para explicar la implementación MPI

La solución MPI se implementó mediante una estrategia de descomposición por filas. El proceso `rank 0` inicializa las matrices completas `A` y `B`. Posteriormente, la matriz `A` se divide en bloques de filas usando `MPI_Scatter`, mientras que la matriz `B` se distribuye completa a todos los procesos mediante `MPI_Bcast`. Cada proceso calcula localmente el bloque correspondiente de la matriz resultado `C`. Finalmente, los bloques parciales de `C` son reunidos en el proceso `rank 0` mediante `MPI_Gather`.

La medición del tiempo se realizó con `MPI_Wtime`, debido a que la ejecución ocurre en una arquitectura distribuida y no resulta conveniente usar mecanismos de medición pensados para una única máquina. Para garantizar una medición más ordenada, se usaron barreras de sincronización mediante `MPI_Barrier` antes y después de la sección medida.

---

## 28. Comandos principales finales

```bash
gcc -O3 -Wall -Wextra matmul_seq.c -o matmul_seq
./matmul_seq 400 4 123456789

mpicc -O3 -Wall -Wextra matmul_mpi.c -o matmul_mpi
./sync_workers.sh

mpirun -np 4 --host head:1,wn1:1,wn2:1,wn3:1 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
mpirun -np 8 --host head:2,wn1:2,wn2:2,wn3:2 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789

./run_seq.sh
./run_mpi.sh
```

---

## 29. Advertencias importantes

1. No usar tamaños demasiado grandes al inicio.
2. Validar primero con `N=400`.
3. Verificar que el checksum sea coherente.
4. Asegurar que el binario exista en todos los nodos.
5. No confundir procesos MPI con hilos.
6. No decir que 8 procesos son 8 nodos.
7. Apagar las instancias cuando no se estén usando.
8. Usar IPs internas para comunicación entre nodos.
9. Usar la misma ruta del binario en todos los nodos.
10. Documentar claramente la arquitectura usada.

---

## 30. Conclusión

La implementación propuesta cumple con el caso de estudio porque usa MPI para paralelizar la multiplicación de matrices sobre un clúster computacional con cuatro nodos en Google Cloud. La estrategia de paralelización se basa en repartir la matriz `A` por filas, distribuir la matriz `B` completa a todos los procesos y reunir los resultados parciales de la matriz `C`.

El experimento permite comparar la versión secuencial ejecutada en `head` contra versiones MPI ejecutadas con 1, 2, 4 y 8 procesos. Con estos resultados se puede calcular speedup, eficiencia y analizar el impacto del número de procesos y del tamaño de matriz sobre el rendimiento.
