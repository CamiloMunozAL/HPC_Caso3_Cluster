# HPC Caso 3 - Multiplicación de Matrices con MPI

> Implementación y evaluación de multiplicación de matrices en un **clúster computacional Google Cloud** con versión secuencial y paralelizada mediante MPI.

---

## Arquitectura del Clúster

El clúster está compuesto por **4 nodos** (instancias `e2-small` en Google Cloud, misma zona):

| Nodo   | Rol                                             |
| ------ | ----------------------------------------------- |
| `head` | Nodo principal (compilación y coordinación MPI) |
| `wn1`  | Nodo trabajador 1                               |
| `wn2`  | Nodo trabajador 2                               |
| `wn3`  | Nodo trabajador 3                               |

---

## Archivos del Proyecto

| Archivo           | Descripción                                       |
| ----------------- | ------------------------------------------------- |
| `matmul_seq.c`    | Multiplicación secuencial de matrices             |
| `matmul_mpi.c`    | Multiplicación paralelizada con MPI               |
| `run_seq.sh`      | Script de pruebas secuenciales                    |
| `run_mpi.sh`      | Script de pruebas MPI                             |
| `sync_workers.sh` | Sincronización de archivos con nodos trabajadores |
| `output/`         | Carpeta de resultados                             |

---

## Configuración Inicial

⚙️ **Nota:** Para configurar el clúster desde cero (SSH, MPI, `/etc/hosts`, usuario `hpcuser`), consulta la [guía paso a paso](guia_hpc_caso3_mpi.md).

Supone que ya tienes:
- 4 instancias Google Cloud configuradas
- SSH sin contraseña entre nodos
- OpenMPI instalado en todos los nodos
- Usuario `hpcuser` común en todos los nodos

---

## Compilación

### Versión Secuencial

```bash
gcc -Wall -Wextra matmul_seq.c -o matmul_seq
```

### Versión MPI

```bash
mpicc -Wall -Wextra matmul_mpi.c -o matmul_mpi
```

Luego, sincroniza el binario con los nodos trabajadores:

```bash
./sync_workers.sh
```

(Todos los nodos deben tener el binario en la misma ruta)

---

## Ejecución

### Versión Secuencial

```bash
./matmul_seq 400 4 123456789
```

### Versión MPI

**Con 4 procesos** (1 por nodo):

```bash
mpirun -np 4 --host head:1,wn1:1,wn2:1,wn3:1 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

**Con 8 procesos** (2 por nodo):

```bash
mpirun -np 8 --host head:2,wn1:2,wn2:2,wn3:2 /home/hpcuser/HPC_Caso3_Cluster/matmul_mpi 400 4 123456789
```

---

## Pruebas de Desempeño

Para automatizar las pruebas, ejecuta los scripts:

```bash
./run_seq.sh  # Pruebas secuenciales
./run_mpi.sh  # Pruebas MPI
```

Los resultados se guardan en:

- `output/output_seq.txt` → Resultados secuenciales
- `output/output_mpi.txt` → Resultados MPI

---

## Estrategia MPI

La versión paralelizada implementa el siguiente patrón:

1. **MPI_Scatter** → Reparte matriz A por filas entre procesos
2. **MPI_Bcast** → Distribuye matriz B completa a todos los procesos
3. **Cómputo Local** → Cada proceso calcula su bloque de C
4. **MPI_Gather** → Reúne resultados parciales en el proceso raíz

**Medición de tiempo:** Se utiliza `MPI_Wtime()` para obtener tiempos consistentes en ambiente distribuido.

---

## Métricas de Desempeño

Con los tiempos obtenidos se calculan:

$$\text{Speedup} = \frac{\text{Tiempo Secuencial}}{\text{Tiempo MPI}}$$

## $$\text{Eficiencia} = \frac{\text{Speedup}}{\text{Número de Procesos}}$$

## Notas Importantes

⚠️ Las pruebas **deben ejecutarse sobre el clúster computacional** y no únicamente en el nodo `head`, para cumplir con el requisito de ejecución distribuida.
