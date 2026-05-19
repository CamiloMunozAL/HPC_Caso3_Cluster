#define _POSIX_C_SOURCE 199309L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/*
  ============================================================
  MULTIPLICACIÓN DE MATRICES CON MPI

  Descripción:
    Esta versión paraleliza la multiplicación de matrices usando MPI.

    A diferencia de la versión con hilos, aquí no se crean pthreads.
    En MPI, el programa se ejecuta varias veces como procesos separados.
    Cada proceso tiene un identificador llamado rank.

  Estrategia:
    - El proceso rank 0 crea las matrices A y B completas.
    - La matriz A se reparte por bloques de filas usando MPI_Scatter.
    - La matriz B se envía completa a todos los procesos usando MPI_Bcast.
    - Cada proceso calcula una parte de la matriz C.
    - Los resultados parciales se reúnen en rank 0 usando MPI_Gather.
    - El tiempo se mide con MPI_Wtime.

  Uso:
    mpirun -np <procesos> --host head,wn1,wn2,wn3 ./matmul_mpi N trials [seed]

  Ejemplos:
    mpirun -np 4 --host head,wn1,wn2,wn3 ./matmul_mpi 400 5
    mpirun -np 4 --host head,wn1,wn2,wn3 ./matmul_mpi 1000 5 123456789

  Salida:
    N processes trial wall_s checksum seed
  ============================================================
*/


// ============================================================
//  PARSEO / UTILIDADES
// ============================================================

static void usage(const char *p) {
    fprintf(stderr, "Uso: %s N trials [seed]\n", p);
    fprintf(stderr, "Ej:  %s 400 5\n", p);
    fprintf(stderr, "Ej:  %s 1000 5 123456789\n", p);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);

    if (!s[0] || (end && *end != '\0')) return 0;
    if (v <= 0 || v > 50000) return 0;

    *out = (int)v;
    return 1;
}


// ============================================================
//  ALEATORIOS / MATRICES
// ============================================================

static int32_t rand_i32(void) {
    uint32_t r = 0;

    r ^= (uint32_t)(rand() & 0x7fff);
    r <<= 15;
    r ^= (uint32_t)(rand() & 0x7fff);
    r <<= 1;
    r ^= (uint32_t)(rand() & 0x0001);

    return (int32_t)r - (int32_t)0x3fffffff;
}

/*
  Llena las matrices A y B.

  Si N == 2, usa un caso pequeño para validar manualmente:

    A = [1 2]
        [3 4]

    B = [5 6]
        [7 8]

    C esperada:
        [19 22]
        [43 50]

  Para otros tamaños, usa valores pseudoaleatorios.
*/
static void fill_random_matrices(int N, int32_t *A, int32_t *B) {
    size_t n = (size_t)N;

    if (N == 2) {
        A[0] = 1; A[1] = 2;
        A[2] = 3; A[3] = 4;

        B[0] = 5; B[1] = 6;
        B[2] = 7; B[3] = 8;
        return;
    }

    for (size_t i = 0; i < n * n; i++) {
        A[i] = rand_i32();
        B[i] = rand_i32();
    }
}

static void zero_block(size_t count, int32_t *C) {
    for (size_t i = 0; i < count; i++) {
        C[i] = 0;
    }
}

static int64_t checksum_matrix(int N, const int32_t *C) {
    size_t n = (size_t)N;
    int64_t s = 0;

    for (size_t i = 0; i < n * n; i++) {
        s += C[i];
    }

    return s;
}


// ============================================================
//  MULTIPLICACIÓN LOCAL
// ============================================================

/*
  Cada proceso ejecuta esta función sobre su propio bloque de filas.

  local_A:
    Bloque de filas de A que recibió este proceso.

  B:
    Matriz B completa. Todos los procesos tienen una copia.

  local_C:
    Bloque de filas de C calculado por este proceso.

  local_rows:
    Número de filas de A que le correspondieron a este proceso.

  Importante:
    Esta función ya no reparte trabajo.
    El reparto lo hace MPI antes, mediante MPI_Scatter.
*/
static void matmul_block(
    int N,
    int local_rows,
    const int32_t *local_A,
    const int32_t *B,
    int32_t *local_C
) {
    size_t n = (size_t)N;

    for (int i = 0; i < local_rows; i++) {
        const int32_t *Ai = &local_A[(size_t)i * n];

        for (int j = 0; j < N; j++) {
            int64_t acc = 0;

            for (int k = 0; k < N; k++) {
                acc += (int64_t)Ai[k] *
                       (int64_t)B[(size_t)k * n + (size_t)j];
            }

            local_C[(size_t)i * n + (size_t)j] = (int32_t)acc;
        }
    }
}


// ============================================================
//  MAIN
// ============================================================

int main(int argc, char **argv) {

    /*
      MPI_Init:
        Inicializa el entorno MPI.

        A partir de aquí, el programa queda bajo control de MPI.
        Todos los procesos lanzados con mpirun ejecutan este mismo main.
    */
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    /*
      MPI_Comm_rank:
        Obtiene el identificador del proceso actual.

        Ejemplo con 4 procesos:
          rank 0
          rank 1
          rank 2
          rank 3

        Normalmente rank 0 se usa como proceso principal o root.
    */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*
      MPI_Comm_size:
        Obtiene el número total de procesos MPI.

        Si ejecutas:
          mpirun -np 4 ./matmul_mpi

        Entonces:
          size = 4
    */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    int trials = 0;
    unsigned seed = 123456789u;

    /*
      En MPI, todos los procesos reciben los argumentos del programa.
      Aun así, para evitar mensajes repetidos, solo rank 0 imprime errores.
    */
    if (argc < 3) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (!parse_int(argv[1], &N)) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (!parse_int(argv[2], &trials)) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (argc >= 4) {
        seed = (unsigned)strtoul(argv[3], NULL, 10);
    }

    /*
      Restricción de esta versión:

      Para usar MPI_Scatter de forma simple, N debe ser divisible
      entre el número de procesos.

      Ejemplo:
        N = 1000, procesos = 4
        1000 / 4 = 250 filas por proceso

      Si N no es divisible entre size, habría que usar MPI_Scatterv,
      que permite repartir cantidades diferentes.
    */
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: N=%d debe ser divisible entre procesos=%d.\n",
                    N, size);
            fprintf(stderr,
                    "Solución: usa un N divisible entre %d o implementa MPI_Scatterv.\n",
                    size);
        }

        MPI_Finalize();
        return 2;
    }

    int local_rows = N / size;

    size_t n = (size_t)N;
    size_t full_elems = n * n;
    size_t local_elems = (size_t)local_rows * n;

    /*
      Matrices completas:
        Solo rank 0 necesita A completa y C completa.

      B completa:
        Todos los procesos necesitan una copia completa de B,
        porque para calcular cualquier fila de C se necesitan todas
        las columnas de B.

      local_A:
        Bloque de filas de A que recibe cada proceso.

      local_C:
        Bloque de filas de C que calcula cada proceso.
    */
    int32_t *A = NULL;
    int32_t *B = NULL;
    int32_t *C = NULL;

    int32_t *local_A = NULL;
    int32_t *local_C = NULL;

    /*
      Todos reservan B completa.
      Todos reservan local_A y local_C.
    */
    B = (int32_t *)malloc(full_elems * sizeof(int32_t));
    local_A = (int32_t *)malloc(local_elems * sizeof(int32_t));
    local_C = (int32_t *)malloc(local_elems * sizeof(int32_t));

    if (!B || !local_A || !local_C) {
        fprintf(stderr,
                "Rank %d: error de memoria en buffers locales.\n",
                rank);

        free(B);
        free(local_A);
        free(local_C);

        MPI_Finalize();
        return 3;
    }

    /*
      Solo rank 0 reserva A y C completas.
      Los demás procesos no necesitan A ni C completas.
    */
    if (rank == 0) {
        A = (int32_t *)malloc(full_elems * sizeof(int32_t));
        C = (int32_t *)malloc(full_elems * sizeof(int32_t));

        if (!A || !C) {
            fprintf(stderr,
                    "Rank 0: error de memoria para matrices completas.\n");

            free(A);
            free(B);
            free(C);
            free(local_A);
            free(local_C);

            MPI_Finalize();
            return 4;
        }

        srand(seed);
        fill_random_matrices(N, A, B);
    }

    /*
      MPI_Bcast:
        Envía la matriz B completa desde rank 0 a todos los procesos.

      Parámetros:
        B             -> buffer que se va a enviar/recibir.
        full_elems    -> cantidad de elementos.
        MPI_INT32_T   -> tipo de dato MPI equivalente a int32_t.
        0             -> proceso root, es decir, quien tiene la información original.
        MPI_COMM_WORLD-> comunicador donde están todos los procesos.

      Después de esta llamada:
        todos los procesos tienen una copia completa de B.
    */
    MPI_Bcast(
        B,
        (int)full_elems,
        MPI_INT32_T,
        0,
        MPI_COMM_WORLD
    );

    /*
      Warm-up no contado:
        Se hace una ejecución previa para reducir efectos de arranque.
        Este tiempo no se reporta.
    */

    /*
      MPI_Scatter:
        Reparte la matriz A entre todos los procesos.

      En esta versión se reparte A por bloques de filas.

      Parámetros de envío:
        A           -> matriz completa en rank 0.
        local_elems -> cantidad de elementos que recibe cada proceso.
        MPI_INT32_T -> tipo de dato.

      Parámetros de recepción:
        local_A     -> buffer local donde cada proceso recibe su bloque.
        local_elems -> cantidad recibida por cada proceso.
        MPI_INT32_T -> tipo de dato.

      Root:
        0 -> rank 0 reparte la matriz A.

      Después de MPI_Scatter:
        cada proceso tiene local_rows filas de A.
    */
    MPI_Scatter(
        A,
        (int)local_elems,
        MPI_INT32_T,
        local_A,
        (int)local_elems,
        MPI_INT32_T,
        0,
        MPI_COMM_WORLD
    );

    zero_block(local_elems, local_C);
    matmul_block(N, local_rows, local_A, B, local_C);

    /*
      MPI_Gather:
        Reúne los bloques local_C calculados por cada proceso
        y los concatena en la matriz C completa dentro de rank 0.

      Después de MPI_Gather:
        rank 0 tiene la matriz C completa.
    */
    MPI_Gather(
        local_C,
        (int)local_elems,
        MPI_INT32_T,
        C,
        (int)local_elems,
        MPI_INT32_T,
        0,
        MPI_COMM_WORLD
    );

    /*
      Encabezado de salida.
      Solo rank 0 imprime, para evitar que todos los procesos escriban lo mismo.
    */
    if (rank == 0) {
        printf("N processes trial wall_s checksum seed\n");
    }

    for (int t = 1; t <= trials; t++) {
        zero_block(local_elems, local_C);

        /*
          MPI_Barrier:
            Sincroniza todos los procesos antes de iniciar la medición.

          Esto evita que algunos procesos empiecen a medir mientras otros
          todavía no están listos.
        */
        MPI_Barrier(MPI_COMM_WORLD);

        /*
          MPI_Wtime:
            Devuelve tiempo de pared compatible con MPI.

          Se usa en lugar de clock_gettime/getrusage porque en MPI pueden
          existir múltiples máquinas, cada una con su propio reloj.
        */
        double wall0 = MPI_Wtime();

        /*
          Repartir nuevamente A en cada trial.
          Esto mantiene la medición incluyendo la comunicación principal
          de distribución de datos.
        */
        MPI_Scatter(
            A,
            (int)local_elems,
            MPI_INT32_T,
            local_A,
            (int)local_elems,
            MPI_INT32_T,
            0,
            MPI_COMM_WORLD
        );

        /*
          Cada proceso calcula su bloque de filas de C.
        */
        matmul_block(N, local_rows, local_A, B, local_C);

        /*
          Reunir los bloques calculados en rank 0.
        */
        MPI_Gather(
            local_C,
            (int)local_elems,
            MPI_INT32_T,
            C,
            (int)local_elems,
            MPI_INT32_T,
            0,
            MPI_COMM_WORLD
        );

        /*
          Segunda barrera:
            Asegura que todos hayan terminado antes de cerrar la medición.
        */
        MPI_Barrier(MPI_COMM_WORLD);

        double wall1 = MPI_Wtime();

        /*
          Solo rank 0 calcula checksum e imprime.
          Los demás procesos no tienen C completa.
        */
        if (rank == 0) {
            int64_t chk = checksum_matrix(N, C);
            double wall_s = wall1 - wall0;

            printf("%d %d %d %.6f %lld %u\n",
                   N,
                   size,
                   t,
                   wall_s,
                   (long long)chk,
                   seed);
        }
    }

    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    /*
      MPI_Finalize:
        Finaliza el entorno MPI.

        Después de esta llamada ya no deben usarse funciones MPI.
    */
    MPI_Finalize();

    return 0;
}