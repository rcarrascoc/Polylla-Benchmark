# PPolylla: Benchmark Comparativo de Algoritmos Paralelos de Polylla

Suite de benchmark y análisis de rendimiento para comparar tres implementaciones del algoritmo **Polylla**:
1. **CPU Secuencial (Baseline)**: [ssalinasfe/Polylla-Mesh-DCEL](https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git)
2. **CPU Multi-core (OpenMP)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git)
3. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)

El sistema genera los datasets de prueba (mallas de triángulos) **directamente en memoria (RAM)** compartida para asegurar que las tres implementaciones evalúen exactamente el mismo conjunto de vértices y caras, admitiendo una semilla (`seed`) para reproducir los experimentos al 100%.

---

## 🚀 Características Principales

- **Submódulos Aislados**: Los repositorios originales se integran como submódulos de Git en el directorio `submodules/`:
  - `submodules/Polylla-Mesh-DCEL` (Baseline secuencial)
  - `submodules/MPolylla` (Paralelo OpenMP)
  - `submodules/GPolylla` (Paralelo CUDA GPU)
  Manteniendo `src/` limpio y desacoplado mediante librerías compartidas independientes (`libseq_runner.so`, `libmpolylla_runner.so`, `libgpolylla_runner.so`).
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
  - **Speedup OpenMP vs Secuencial Baseline**
  - **Speedup GPU vs OpenMP Multi-Core**
- **Ejecución con un solo comando**: Script `./run_benchmark.sh` y CLI `scripts/benchmark.py` con soporte para barridos (*sweeps*) automáticos, tablas CSV y generación de gráficos con Matplotlib.

---

## 📁 Estructura del Proyecto

```
PPolylla/
├── CMakeLists.txt           # Configuración de compilación CMake (C++14, CUDA sm_89/native, OpenMP)
├── run_benchmark.sh         # Script ejecutable principal de benchmark
├── README.md                # Documentación del proyecto
├── submodules/              # Submódulos externos (repositorios remotos)
│   ├── Polylla-Mesh-DCEL/   # Baseline secuencial en CPU
│   ├── MPolylla/            # Implementación CPU (OpenMP)
│   └── GPolylla/            # Implementación GPU (CUDA)
├── src/                     # Código fuente de la suite de benchmark
│   ├── common/
│   │   └── benchmark_types.hpp # Estructuras de datos, timers, reporte y serialización
│   ├── generators/
│   │   ├── delaunator.hpp      # Triangulador de Delaunay 2D ultra-rápido en C++
│   │   └── mesh_generator.hpp  # Generadores en memoria de mallas (grid y uniform)
│   ├── runners/
│   │   ├── seq_runner.hpp / .cpp# Envoltorio de Polylla-Mesh-DCEL en libseq_runner.so
│   │   ├── cpu_runner.hpp / .cpp# Envoltorio de MPolylla en libmpolylla_runner.so
│   │   └── gpu_runner.hpp / .cu # Envoltorio de GPolylla en libgpolylla_runner.so
│   └── main.cpp             # Ejecutable principal C++ de benchmark
├── scripts/
│   └── benchmark.py         # Orquestador Python para barridos (sweeps) y gráficos
├── results/                 # Resultados de benchmarks en JSON y CSV
└── plots/                   # Gráficos generados (PNG)
```

---

## 🖥️ Uso Rápido (Un Solo Comando)

### 1. Benchmark Individual Completo (Secuencial vs OpenMP vs GPU)

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
- Reportes individuales JSON en `results/`
- Tabla consolidada CSV en `results/sweep_uniform_seed42.csv`
- Gráfico de Speedup en `plots/speedup_uniform.png` (curvas GPU vs Seq, OpenMP vs Seq, GPU vs OpenMP)
- Gráfico de Tiempo en `plots/time_uniform.png` (curvas de los 3 códigos en escala logarítmica)

---

## 📊 Ejemplo de Salida en Terminal

```
===================================================================================================================
                                 POLYLLA BENCHMARK COMPARISON REPORT                                               
                 (Sequential Baseline vs OpenMP Multi-Core CPU vs CUDA GPU)                                       
===================================================================================================================
  Mesh Type:       uniform
  Requested Size:  100000
  Random Seed:     42
  CPU Threads:     20
  Mesh Gen Time:   137.779 ms
-------------------------------------------------------------------------------------------------------------------
  Metric / Element             SEQ (Baseline)       CPU (MPolylla)       GPU (GPolylla)       Validation           
-------------------------------------------------------------------------------------------------------------------
  Vertices (n_vertices)                 100002               100002               100002          MATCH (CPU=GPU)
  Faces (n_faces)                       199992               199992               199992          MATCH (CPU=GPU)
  Half-Edges (n_half_edges)             599986               599986               599986          MATCH (CPU=GPU)
  Polygons (n_polygons)                  32046                32003                32053         DIFF (CPU!=GPU)*
-------------------------------------------------------------------------------------------------------------------
  Phase Timers (ms)            SEQ (Baseline)       CPU (MPolylla)       GPU (GPolylla)      Speedup GPU/Seq       
-------------------------------------------------------------------------------------------------------------------
  1. Triangulation Struct             165.0390             156.4100             165.4390                  1.00x
  2. Label Max Edges                   42.3635               5.7677               4.7541                  8.91x
  3. Label Frontier Edges              14.7088               1.0968               0.3776                 38.96x
  4. Label Seed Edges                  16.4518               1.0951               0.0000                     -
  5. Repair Extra Frontier              2.1364               1.4482               0.0468                 45.64x
  6. Traversal Phase 1                 16.4800               3.1899               0.0443                371.77x
  7. Traversal Phase 2 (Search)             0.0000               1.8583               0.0191                     -
  8. Repair Overwrite Seed              0.0000               1.4190               0.0405                     -
  9. Label Seed Scan                    0.0000               0.0000               0.0000                     -
  10. Seed Compaction                   0.0000               0.5099               0.1905                     -
-------------------------------------------------------------------------------------------------------------------
  TOTAL ALGORITHM TIME                 92.1405              16.3848               5.1317                 17.96x
-------------------------------------------------------------------------------------------------------------------
  GPU Transfer Host->Device:                 1.7640 ms
  GPU Transfer Device->Host:                 2.1068 ms
  GPU TOTAL (Alg + Transfers):               9.0025 ms
-------------------------------------------------------------------------------------------------------------------
  >>> SPEEDUP GPU vs Sequential (Alg only):             17.96x
  >>> SPEEDUP GPU vs Sequential (With Transfers):       10.23x
  >>> SPEEDUP OpenMP vs Sequential:                      5.62x
  >>> SPEEDUP GPU vs OpenMP (Alg only):                  3.19x
===================================================================================================================
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
