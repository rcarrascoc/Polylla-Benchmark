# PPolylla: Benchmark Comparativo de Algoritmos Paralelos de Polylla

Suite de benchmark y análisis de rendimiento para comparar las implementaciones paralelas del algoritmo **Polylla**:
1. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)
2. **CPU Multi-core (OpenMP)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git)

El sistema genera los datasets de prueba (mallas de triángulos) **directamente en memoria (RAM)** compartida para asegurar que ambas implementaciones evalúen exactamente el mismo conjunto de vértices y caras, admitiendo una semilla (`seed`) para reproducir los experimentos al 100%.

---

## 🚀 Características Principales

- **Submódulos Aislados**: Los repositorios originales se integran como submódulos de Git en el directorio `submodules/` (`submodules/GPolylla` y `submodules/MPolylla`), manteniendo `src/` limpio y desacoplado.
- **Generación en Memoria (Zero Disk Overhead)**:
  - **Grilla (`grid`)**: Malla estructurada regular de $k \times k$ vértices con división diagonal idéntica a la usada en las publicaciones de GPolylla.
  - **Uniforme (`uniform`)**: Puntos distribuidos uniformemente en $[0, 1] \times [0, 1]$ con ajuste a los bordes (`move_point`) y triangulación de Delaunay en memoria en apenas milisegundos gracias a `delaunator-cpp`.
- **Reproducibilidad Exacta**: Parámetro `--seed` para replicar con precisión las distribuciones uniformes.
- **Rescate y Desglose Completo de Timers**:
  1. Construcción de estructura de datos de triangulación (`time_triangulation_generation`)
  2. Etiquetado de aristas máximas (`d_time_to_label_max_edges`)
  3. Etiquetado de aristas frontera (`d_time_to_label_frontier_edges`)
  4. Etiquetado de aristas semilla (`d_time_to_label_seed_edges`)
  5. Reparación de aristas frontera adicionales (`d_time_to_label_extra_frontier_edge`)
  6. Recorrido / Traversal fase 1 (`d_time_to_traversal`)
  7. Recorrido fase 2 / Búsqueda frontera (`d_time_to_traversal_search_frontier_edge`)
  8. Sobrescritura de semillas (`d_time_to_overwrite_seed`)
  9. Escaneo prefijo de semillas (`d_time_to_label_scan_edges`)
  10. Compactación de semillas (`d_time_to_label_compaction_edges`)
  - Transferencias host-device (`d_time_copy_to_device`, `d_time_to_back_to_host`)
  - **Tiempo Total de Algoritmo** y **Speedup** (GPU vs CPU, con y sin transferencias).
- **Ejecución con un solo comando**: Script `./run_benchmark.sh` y CLI `scripts/benchmark.py` con soporte para barridos (*sweeps*) automáticos, tablas CSV y generación de gráficos con Matplotlib.

---

## 📁 Estructura del Proyecto

```
PPolylla/
├── CMakeLists.txt           # Configuración de compilación CMake (C++14, CUDA sm_89/native, OpenMP)
├── run_benchmark.sh         # Script ejecutable principal de benchmark
├── README.md                # Documentación del proyecto
├── submodules/              # Submódulos externos (repositorios remotos)
│   ├── GPolylla/            # Implementación GPU (CUDA)
│   └── MPolylla/            # Implementación CPU (OpenMP)
├── src/                     # Código fuente de la suite de benchmark
│   ├── common/
│   │   └── benchmark_types.hpp # Estructuras de datos, timers, reporte y serialización
│   ├── generators/
│   │   ├── delaunator.hpp      # Triangulador de Delaunay 2D ultra-rápido en C++
│   │   └── mesh_generator.hpp  # Generadores en memoria de mallas (grid y uniform)
│   ├── runners/
│   │   ├── gpu_runner.hpp / .cu # Envoltorio de GPolylla aislado en libgpolylla_runner.so
│   │   └── cpu_runner.hpp / .cpp# Envoltorio de MPolylla aislado en libmpolylla_runner.so
│   └── main.cpp             # Ejecutable principal C++ de benchmark
├── scripts/
│   └── benchmark.py         # Orquestador Python para barridos (sweeps) y gráficos
├── results/                 # Resultados de benchmarks en JSON y CSV
└── plots/                   # Gráficos generados (PNG)
```

---

## 🛠️ Requisitos del Sistema

- **Compilador C++**: GCC / G++ 11+ con soporte para OpenMP (`-fopenmp`)
- **CUDA Toolkit**: NVCC 12.0+ con soporte para arquitectura GPU (e.g. sm_89, sm_80)
- **GPU NVIDIA**: Compatible con CUDA (e.g. RTX 4090, RTX 3090, A100, etc.)
- **CMake**: Versión 3.18 o superior
- **Python 3**: Con `pandas` y `matplotlib` para gráficos y barridos

---

## ⚙️ Compilación

El proyecto se compila automáticamente al ejecutar `./run_benchmark.sh`, o manualmente con:

```bash
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
make -j$(nproc)
cd ..
```

---

## 🖥️ Uso Rápido (Un Solo Comando)

### 1. Benchmark Individual (Malla Uniforme con Semilla)

```bash
./run_benchmark.sh --type uniform --size 100000 --seed 42
```

### 2. Benchmark Individual (Malla de Grilla Regular)

```bash
./run_benchmark.sh --type grid --size 1000000
```

### 3. Benchmark sobre un Archivo `.off` Existente

```bash
./run_benchmark.sh --off submodules/MPolylla/examples/pikachu.off --threads 16
```

### 4. Barrido Automático de Tamaños y Generación de Gráficos

Para evaluar múltiples tamaños (ej. 10k, 50k, 100k, 500k, 1M de vértices):

```bash
./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000,500000,1000000 --seed 42
```

Esto generará automáticamente:
- Reportes individuales JSON en `results/`
- Tabla consolidada CSV en `results/sweep_uniform_seed42.csv`
- Gráfico de Speedup en `plots/speedup_uniform.png`
- Gráfico de Tiempo de Ejecución en `plots/time_uniform.png`

---

## 📊 Ejemplo de Salida en Terminal

```
========================================================================================
                         POLYLLA BENCHMARK COMPARISON REPORT                            
========================================================================================
  Mesh Type:       uniform
  Requested Size:  100000
  Random Seed:     42
  CPU Threads:     20
  Mesh Gen Time:   132.327 ms
----------------------------------------------------------------------------------------
  Metric / Element                GPU (GPolylla)       CPU (MPolylla)       Difference  
----------------------------------------------------------------------------------------
  Vertices (n_vertices)                       100002              100002            MATCH
  Faces (n_faces)                             199992              199992            MATCH
  Half-Edges (n_half_edges)                   599986              599986            MATCH
  Polygons (n_polygons)                        32053               32004               49
----------------------------------------------------------------------------------------
  Phase Timers (ms)               GPU (GPolylla)       CPU (MPolylla)          Speedup  
----------------------------------------------------------------------------------------
  1. Triangulation Data Struct              164.1250            164.4400            1.00x
  2. Label Max Edges                          4.7406              6.1351            1.29x
  3. Label Frontier Edges                     0.3997              1.2112            3.03x
  4. Label Seed Edges                         0.0000              1.9395               -
  5. Repair Extra Frontier                    0.0444              2.0526           46.24x
  6. Traversal Phase 1                        0.0427              2.6418           61.87x
  7. Traversal Phase 2 (Search)               0.0201              0.5861           29.16x
  8. Repair Overwrite Seed                    0.0406              0.7827           19.27x
  9. Label Seed Scan                          0.0000              0.0000               -
  10. Seed Compaction                         0.1734              0.5034            2.90x
----------------------------------------------------------------------------------------
  TOTAL ALGORITHM TIME                        5.1403             15.8524            3.08x
----------------------------------------------------------------------------------------
  GPU Transfer To Device:                     1.7528 ms
  GPU Transfer Back To Host:                  1.8392 ms
  GPU TOTAL (Alg + Transfers):                8.7323 ms
----------------------------------------------------------------------------------------
  >>> SPEEDUP (Algorithm only):                 3.08x
  >>> SPEEDUP (With Transfers):                 1.82x
========================================================================================
```

---

## 🔧 Opciones Disponibles de la Línea de Comandos

| Opción | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `-t, --type <grid\|uniform>` | Tipo de malla a generar en memoria | `uniform` |
| `-s, --size <N>` | Cantidad de vértices solicitados | `100000` |
| `-r, --seed <SEED>` | Semilla del generador pseudoaleatorio | `42` |
| `-j, --threads <T>` | Hilos OpenMP para CPU (0 = todos los núcleos detectados) | `0` |
| `-o, --output <PATH>` | Ruta del archivo JSON de salida | `results/benchmark_...json` |
| `-f, --off <FILE>` | Usar un archivo `.off` existente en lugar de generar en memoria | Ninguno |
| `--save-mesh <PATH>` | Guarda la malla generada en un archivo `.off` persistente | Ninguno |
| `--sweep` | Activa el modo barrido múltiple a través del script Python | `false` |
| `--sizes <N1,N2,...>` | Lista separada por comas de tamaños para el barrido | `10k,50k,100k,500k,1M` |
| `-h, --help` | Muestra el menú de ayuda | - |

---

## 🔬 Reproducibilidad de Experimentos

Para replicar un experimento en cualquier momento o máquina con la misma configuración:

```bash
# Ejemplo: reproducir exactamente con semilla 12345
./run_benchmark.sh --type uniform --size 500000 --seed 12345
```
El archivo de resultados se guardará en `results/benchmark_uniform_500000_seed12345.json` con todos los timestamps, métricas de polígonos, aristas y timers.
