# PPolylla: Benchmark Comparativo de Algoritmos Paralelos de Polylla

Suite de benchmark y análisis de rendimiento para comparar cuatro implementaciones del algoritmo **Polylla**:
1. **CPU Secuencial (Baseline)**: [ssalinasfe/Polylla-Mesh-DCEL](https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git)
2. **CPU Multi-core (OpenMP Loops)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (rama `main`)
3. **CPU Multi-core (OpenMP Tasks)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (rama `Task`)
4. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)

El sistema genera los datasets de prueba (mallas de triángulos) **directamente en memoria (RAM)** compartida para asegurar que las cuatro implementaciones evalúen exactamente el mismo conjunto de vértices y caras, admitiendo una semilla (`seed`) para reproducir los experimentos al 100%.

---

## 🚀 Características Principales

- **Submódulos Aislados**: Los repositorios originales se integran como submódulos de Git en el directorio `submodules/`:
  - `submodules/Polylla-Mesh-DCEL` (Baseline secuencial)
  - `submodules/MPolylla` (Paralelo OpenMP Loops - rama `main`)
  - `submodules/MPolylla-Task` (Paralelo OpenMP Tasks - rama `Task`)
  - `submodules/GPolylla` (Paralelo CUDA GPU)
  Manteniendo `src/` limpio y desacoplado mediante librerías compartidas independientes (`libseq_runner.so`, `libmpolylla_runner.so`, `libmpolylla_task_runner.so`, `libgpolylla_runner.so`).
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
  - **Speedup GPU vs Secuencial Baseline** (con y sin transferencias)
  - **Speedup OpenMP (Loops) vs Secuencial Baseline**
  - **Speedup OpenMP (Tasks) vs Secuencial Baseline**
  - **Speedup OpenMP (Tasks) vs OpenMP (Loops)**
  - **Speedup GPU vs OpenMP (Loops y Tasks)**
- **Exportación Automática en `.txt` y `.json`**: Genera un informe legible con metadata exhaustiva de hardware (CPU, GPU, RAM, OS, CUDA), commits exactos de cada submódulo y parámetros de ejecución.
- **Ejecución con un solo comando**: Script `./run_benchmark.sh` y CLI `scripts/benchmark.py` con soporte para barridos (*sweeps*) automáticos, tablas CSV y generación de gráficos con Matplotlib.

---

## 📁 Estructura del Proyecto

```
PPolylla/
├── CMakeLists.txt              # Configuración de compilación CMake (C++14, CUDA sm_89/native, OpenMP)
├── run_benchmark.sh            # Script ejecutable principal de benchmark
├── README.md                   # Documentación del proyecto
├── submodules/                 # Submódulos externos (repositorios remotos)
│   ├── Polylla-Mesh-DCEL/      # Baseline secuencial en CPU
│   ├── MPolylla/               # Implementación CPU (OpenMP Loops - rama main)
│   ├── MPolylla-Task/          # Implementación CPU (OpenMP Tasks - rama Task)
│   └── GPolylla/               # Implementación GPU (CUDA)
├── src/                        # Código fuente de la suite de benchmark
│   ├── common/
│   │   └── benchmark_types.hpp # Estructuras de datos, timers, metadata, TXT y JSON
│   ├── generators/
│   │   ├── delaunator.hpp         # Triangulador de Delaunay 2D ultra-rápido en C++
│   │   └── mesh_generator.hpp     # Generadores en memoria de mallas (grid y uniform)
│   ├── runners/
│   │   ├── seq_runner.hpp / .cpp     # Envoltorio de Polylla-Mesh-DCEL en libseq_runner.so
│   │   ├── cpu_runner.hpp / .cpp     # Envoltorio de MPolylla Loops en libmpolylla_runner.so
│   │   ├── cpu_task_runner.hpp / .cpp# Envoltorio de MPolylla Tasks en libmpolylla_task_runner.so
│   │   └── gpu_runner.hpp / .cu    # Envoltorio de GPolylla en libgpolylla_runner.so
│   └── main.cpp                # Ejecutable principal C++ de benchmark
├── scripts/
│   └── benchmark.py            # Orquestador Python para barridos (sweeps) y gráficos
├── results/                    # Resultados de benchmarks en TXT, JSON y CSV
└── plots/                      # Gráficos generados (PNG)
```

---

## 🖥️ Uso Rápido (Un Solo Comando)

### 1. Benchmark Individual Completo (Secuencial vs OpenMP Loops vs Tasks vs GPU)

```bash
# Malla uniforme de 100,000 vértices con semilla 42
./run_benchmark.sh --type uniform --size 100000 --seed 42

# Malla de grilla de 1,000,000 puntos (omitiendo secuencial para acelerar)
./run_benchmark.sh --type grid --size 1000000 --no-seq
```

### 2. Barrido Automático de Tamaños y Gráficos

Para evaluar múltiples tamaños (ej. 10k, 50k, 100k vértices):

```bash
./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000 --seed 42
```

Salida generada:
- Reportes individuales legibles en `results/*.txt` y estructurados en `results/*.json`
- Tabla consolidada CSV en `results/sweep_uniform_seed42.csv`
- Gráfico de Speedup en `plots/speedup_uniform.png`
- Gráfico de Tiempo en `plots/time_uniform.png` (curvas en escala logarítmica)

---

## 📊 Ejemplo de Salida en Terminal

```
======================================================================================================================================
                                                  POLYLLA BENCHMARK COMPARISON REPORT                                                 
                              (Sequential Baseline vs OpenMP Loops vs OpenMP Tasks vs CUDA GPU)                                       
======================================================================================================================================
  Mesh Type:       uniform
  Requested Size:  100000
  Random Seed:     42
  CPU Threads:     20
  Mesh Gen Time:   139.476 ms
--------------------------------------------------------------------------------------------------------------------------------------
  Metric / Element             SEQ (Baseline)      OMP (Loops)       OMP (Tasks)       GPU (GPolylla)           Validation            
--------------------------------------------------------------------------------------------------------------------------------------
  Vertices (n_vertices)                100002            100002            100002              100002        MATCH (All Parallel)
  Faces (n_faces)                      199992            199992            199992              199992        MATCH (All Parallel)
  Half-Edges (n_half_edges)            599986            599986            599986              599986        MATCH (All Parallel)
  Polygons (n_polygons)                 32046             32004             32001               32053            DIFF (Parallel)*
--------------------------------------------------------------------------------------------------------------------------------------
  Phase Timers (ms)            SEQ (Baseline)      OMP (Loops)       OMP (Tasks)       GPU (GPolylla)         Speedup GPU/Seq         
--------------------------------------------------------------------------------------------------------------------------------------
  1. Triangulation Struct            156.5320          161.9130          161.9700            158.3400                     0.99x
  2. Label Max Edges                  43.7306            3.9194            2.3554              4.7260                     9.25x
  3. Label Frontier Edges             14.8783            1.1791            1.0246              0.3735                    39.83x
  4. Label Seed Edges                 15.9823            1.2180            0.9418              0.0000                         -
  5. Repair Extra Frontier             2.1867            1.4740            1.1416              0.0442                    49.47x
  6. Traversal Phase 1                16.7280            2.3387            2.2993              0.0435                   384.40x
  7. Traversal Phase 2 (Search)            0.0000            0.4919            0.5811              0.0197                         -
  8. Repair Overwrite Seed             0.0000            0.7310            0.8704              0.0401                         -
  9. Label Seed Scan                   0.0000            0.0000            0.0000              0.0000                         -
  10. Seed Compaction                  0.0000            0.4815            0.5413              0.1783                         -
--------------------------------------------------------------------------------------------------------------------------------------
  TOTAL ALGORITHM TIME                93.5059           11.8335            9.7556              5.0995                    18.34x
--------------------------------------------------------------------------------------------------------------------------------------
  GPU Transfer Host->Device:                1.7028 ms
  GPU Transfer Device->Host:                2.0751 ms
  GPU TOTAL (Alg + Transfers):              8.8774 ms
--------------------------------------------------------------------------------------------------------------------------------------
  >>> SPEEDUP GPU vs Sequential (Alg only):                 18.34x
  >>> SPEEDUP GPU vs Sequential (With Transfers):           10.53x
  >>> SPEEDUP OpenMP (Loops) vs Sequential:                  7.90x
  >>> SPEEDUP OpenMP (Tasks) vs Sequential:                  9.58x
  >>> SPEEDUP GPU vs OpenMP (Loops):                         2.32x
  >>> SPEEDUP GPU vs OpenMP (Tasks):                         1.91x
  >>> SPEEDUP OpenMP (Tasks) vs OpenMP (Loops):              1.21x
======================================================================================================================================
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
| `--no-seq` | Omitir el baseline secuencial en CPU | `false` (ejecuta los 3) |
| `--save-mesh <PATH>` | Guarda la malla generada en un archivo `.off` persistente | Ninguno |
| `--sweep` | Activa el modo barrido múltiple a través del script Python | `false` |
| `--sizes <N1,N2,...>` | Lista separada por comas de tamaños para el barrido | `10k,50k,100k,500k,1M` |
| `-h, --help` | Muestra el menú de ayuda | - |
