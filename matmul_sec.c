#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

/*
  ============================================================
  MULTIPLICACIÓN DE MATRICES - VERSIÓN SECUENCIAL

  Descripción:
    Este programa ejecuta la multiplicación de dos matrices cuadradas
    A y B de tamaño N x N, almacenando el resultado en la matriz C.

    Esta versión es secuencial y sirve como línea base para comparar
    posteriormente contra una versión paralelizada con MPI.

  Uso:
    ./matmul_seq N trials [seed]

  Ejemplos:
    ./matmul_seq 400 5
    ./matmul_seq 1000 5 123456789

  Salida:
    N version trial wall_s user_s kernel_s cpu_total_s checksum seed

  Donde:
    N            = tamaño de la matriz N x N
    version      = tipo de ejecución, en este caso "seq"
    trial        = número de repetición
    wall_s       = tiempo real transcurrido en segundos
    user_s       = tiempo CPU en modo usuario
    kernel_s     = tiempo CPU en modo kernel
    cpu_total_s  = user_s + kernel_s
    checksum     = suma de los elementos de C, usada como verificación básica
    seed         = semilla usada para generar A y B
  ============================================================
*/


// ============================================================
//  MEDICIÓN DE TIEMPO REAL: WALL TIME
// ============================================================

/*
  Devuelve el tiempo real actual en segundos.

  Se usa CLOCK_MONOTONIC porque no se ve afectado por cambios del reloj
  del sistema. Esta medición representa el tiempo transcurrido observado
  por el usuario, es decir, el "wall time".
*/
static double wall_seconds_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}


// ============================================================
//  MEDICIÓN DE TIEMPO DE CPU DEL PROCESO
// ============================================================
static int process_cpu_seconds(double *user_s, double *kernel_s) {
  struct rusage ru;

  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    *user_s = 0.0;
    *kernel_s = 0.0;
    return 0;
  }

  *user_s =
      (double)ru.ru_utime.tv_sec +
      (double)ru.ru_utime.tv_usec * 1e-6;

  *kernel_s =
      (double)ru.ru_stime.tv_sec +
      (double)ru.ru_stime.tv_usec * 1e-6;

  return 1;
}


// ============================================================
//  PARSEO DE ARGUMENTOS
// ============================================================

static void usage(const char *program_name) {
  fprintf(stderr, "Uso: %s N trials [seed]\n", program_name);
  fprintf(stderr, "Ej:  %s 400 5\n", program_name);
  fprintf(stderr, "Ej:  %s 1000 5 123456789\n", program_name);
}

/*
  Convierte una cadena a entero positivo.

  Se usa para validar N y trials.
*/
static int parse_int(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);

  if (!s[0] || (end && *end != '\0')) {
    return 0;
  }

  if (v <= 0 || v > 50000) {
    return 0;
  }

  *out = (int)v;
  return 1;
}


// ============================================================
//  GENERACIÓN DE DATOS
// ============================================================

/*
  Genera un entero pseudoaleatorio de 32 bits con signo.

  Se combinan varias llamadas a rand() para construir un valor de mayor
  tamaño, ya que rand() no siempre produce directamente 32 bits útiles.

  El rango se desplaza para obtener valores positivos y negativos.
*/
static int32_t rand_i32(void) {
  uint32_t r = 0;

  r ^= (uint32_t)(rand() & 0x7fff);
  r <<= 15;
  r ^= (uint32_t)(rand() & 0x7fff);
  r <<= 1;
  r ^= (uint32_t)(rand() & 0x0001);

  return (int32_t)(r) - (int32_t)0x3fffffff;
}

/*
  Llena las matrices A y B con números pseudoaleatorios.

  Ambas matrices tienen tamaño N x N y están almacenadas en memoria lineal.
  La posición [i][j] se accede como:

    matriz[i * N + j]
*/
static void fill_random_matrices(int N, int32_t *A, int32_t *B) {
  size_t n = (size_t)N;
  size_t total = n * n;

  if (N == 2) {
    A[0] = 1; A[1] = 2;
    A[2] = 3; A[3] = 4;

    B[0] = 5; B[1] = 6;
    B[2] = 7; B[3] = 8;
    return;
  }

  for (size_t i = 0; i < total; i++) {
    A[i] = rand_i32();
    B[i] = rand_i32();
  }
}

/*
  Inicializa la matriz C en cero antes de cada multiplicación.
*/
static void zero_matrix(int N, int32_t *C) {
  size_t n = (size_t)N;
  size_t total = n * n;

  for (size_t i = 0; i < total; i++) {
    C[i] = 0;
  }
}


// ============================================================
//  RESERVA DE MEMORIA
// ============================================================

/*
  Reserva memoria dinámica para las matrices A, B y C.

  Cada matriz tiene N*N elementos de tipo int32_t.
  Cada elemento ocupa 4 bytes.

  Memoria aproximada por matriz:
    bytes = N * N * sizeof(int32_t)

  Memoria total aproximada:
    3 * N * N * sizeof(int32_t)
*/
static int allocate_matrices(int N, int32_t **A, int32_t **B, int32_t **C) {
  size_t n = (size_t)N;
  size_t total = n * n;
  size_t bytes = total * sizeof(int32_t);

  *A = (int32_t *)malloc(bytes);
  *B = (int32_t *)malloc(bytes);
  *C = (int32_t *)malloc(bytes);

  if (!(*A) || !(*B) || !(*C)) {
    free(*A);
    free(*B);
    free(*C);

    *A = NULL;
    *B = NULL;
    *C = NULL;

    return 0;
  }

  return 1;
}


// ============================================================
//  CHECKSUM
// ============================================================

/*
  Calcula una suma simple de todos los elementos de C.

  No es una verificación matemática completa, pero sirve para detectar
  cambios evidentes entre ejecuciones o versiones del algoritmo.
*/
static int64_t checksum_matrix(int N, const int32_t *C) {
  size_t n = (size_t)N;
  size_t total = n * n;
  int64_t checksum = 0;

  for (size_t i = 0; i < total; i++) {
    checksum += C[i];
  }

  return checksum;
}


// ============================================================
//  MULTIPLICACIÓN DE MATRICES
// ============================================================

/*
  Multiplica A x B y almacena el resultado en C.

  Las matrices están almacenadas en formato lineal por filas.

  Fórmula:
    C[i][j] = sumatoria de A[i][k] * B[k][j]

  Nota:
    Se usa int64_t para el acumulador interno con el fin de reducir
    el riesgo de desbordamiento durante la suma de productos.
    Al final, el resultado se convierte nuevamente a int32_t.
*/
static void matmul(int N, const int32_t *A, const int32_t *B, int32_t *C) {
  size_t n = (size_t)N;

  for (int i = 0; i < N; i++) {
    const int32_t *Ai = &A[(size_t)i * n];

    for (int j = 0; j < N; j++) {
      int64_t acc = 0;

      for (int k = 0; k < N; k++) {
        acc += (int64_t)Ai[k] *
               (int64_t)B[(size_t)k * n + (size_t)j];
      }

      C[(size_t)i * n + (size_t)j] = (int32_t)acc;
    }
  }
}


// ============================================================
//  MAIN
// ============================================================

int main(int argc, char **argv) {
  int N = 0;
  int trials = 0;
  unsigned seed = 123456789u;

  /*
    Argumentos esperados:
      argv[1] = N
      argv[2] = trials
      argv[3] = seed opcional
  */
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  if (!parse_int(argv[1], &N)) {
    usage(argv[0]);
    return 1;
  }

  if (!parse_int(argv[2], &trials)) {
    usage(argv[0]);
    return 1;
  }

  if (argc >= 4) {
    seed = (unsigned)strtoul(argv[3], NULL, 10);
  }

  srand(seed);

  int32_t *A = NULL;
  int32_t *B = NULL;
  int32_t *C = NULL;

  if (!allocate_matrices(N, &A, &B, &C)) {
    size_t n = (size_t)N;
    size_t bytes = n * n * sizeof(int32_t);

    fprintf(stderr,
            "Error: memoria insuficiente para N=%d "
            "(%.2f MB por matriz, %.2f MB total aprox.)\n",
            N,
            (double)bytes / (1024.0 * 1024.0),
            (double)(3 * bytes) / (1024.0 * 1024.0));

    return 2;
  }

  /*
    Se generan A y B una sola vez para que todos los trials usen
    los mismos datos de entrada.
  */
  fill_random_matrices(N, A, B);

  /*
    Warm-up:
    Ejecución inicial no contada.
    Sirve para reducir efectos iniciales de carga, caché o estado frío.
  */
  zero_matrix(N, C);
  matmul(N, A, B, C);

  /*
    Encabezado CSV-like.
    Se imprime una vez para facilitar análisis posterior.
  */
  printf("N version trial wall_s user_s kernel_s cpu_total_s checksum seed\n");

  for (int t = 1; t <= trials; t++) {
    zero_matrix(N, C);

    double user0 = 0.0;
    double kernel0 = 0.0;
    double user1 = 0.0;
    double kernel1 = 0.0;

    /*
      Medición antes de la multiplicación.
    */
    double wall0 = wall_seconds_now();
    process_cpu_seconds(&user0, &kernel0);

    /*
      Sección medida:
      multiplicación secuencial de matrices.
    */
    matmul(N, A, B, C);

    /*
      Medición después de la multiplicación.
    */
    process_cpu_seconds(&user1, &kernel1);
    double wall1 = wall_seconds_now();

    int64_t checksum = checksum_matrix(N, C);

    double wall_s = wall1 - wall0;
    double user_s = user1 - user0;
    double kernel_s = kernel1 - kernel0;
    double cpu_total_s = user_s + kernel_s;

    /*
      Salida:
        N version trial wall_s user_s kernel_s cpu_total_s checksum seed
    */
    printf("%d seq %d %.6f %.6f %.6f %.6f %lld %u\n",
           N,
           t,
           wall_s,
           user_s,
           kernel_s,
           cpu_total_s,
           (long long)checksum,
           seed);
  }

  free(A);
  free(B);
  free(C);

  return 0;
}