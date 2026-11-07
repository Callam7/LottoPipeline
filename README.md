# Lotto Generator

** Author’s Note **

Before anything else: this project does **not** guarantee lottery predictions. Lotto NZ is a heavily regulated system, independently audited, and engineered to be fair at all times. Nothing in this repository overrides that fact, and nothing here should be treated as a promise of “winning numbers.” This project exists as a technical experiment — not a loophole.

It started as a personal challenge to the idea that Lotto is “just random.” The rules never change: main numbers 1–40, Powerball 1–10. That makes it pseudo-random rather than chaotic. If anything meaningful exists in the structure, it should be measurable; if nothing exists, proving that is valuable too. What began as a small script grew into a full analytical and machine-learning pipeline designed to test structure, extract behavior, and see how far adaptive modelling can be pushed against a system intentionally designed to resist prediction.

Over time the project shifted from “generate numbers” to a much larger goal: building the foundations of a self-adapting model — something capable of training on its own history, monitoring its performance, and modifying its own internal logic when needed. Lotto simply provides a stable sandbox to explore those ideas.

As for the tools behind it: Python is my preferred language and the one I’m most fluent in. I know enough about other languages to an extent, such as C#, C++, Kotlin, SQL, and typescript, but Python gives me just a bit more flexibility, clarity, and ecosystem support required for something this experimental.

---

## Purpose

The **Lotto Generator** is a Python application that analyzes historical lottery data and generates tickets based on multiple statistical and machine learning methods.

It uses a pipeline of “pipes” — statistical analysis, clustering, Monte Carlo simulation, Markov chains, Shannon entropy, Bayesian fusion with mechanics estimation, quantum-enhanced features, and deep learning. Each pipe produces features from the historical draws, which are combined and used to train the deep learning model.

The long-term goal is for the pipeline to update and re-train itself automatically as new draws come in, with the aim of improving accuracy while avoiding overfitting. For now, Lotto is just the sandbox for testing these methods.

---

## Table of Contents
1. [Installation Instructions](#installation-instructions)  
2. [Pipeline Explanation](#pipeline-explanation)  
3. [Future Additions](#future-additions)  

---

## Installation Instructions

To set up the project locally, follow these step-by-step instructions for your operating system. The process includes installing the required Python version, necessary packages, and configuring specific dependencies for TensorFlow, Pennylane (quantum), and SQLite.

**Important compatibility note**: This project is currently only tested and guaranteed to run with **Python 3.12** (use `py -3.12 main.py` on Windows, `python3.12 main.py` on Linux/macOS). TensorFlow and Pennylane versions in `requirements.txt` are compatible only with Python 3.12.

### Prerequisites
1. **Python**: Python 3.12 (required).  
2. **Pip**: The Python package manager.  
3. **SQLite**: Must be installed and in PATH.  
4. **TensorFlow + Pennylane**: For deep learning and quantum layers.

---

### **Windows Installation**
#### 1. Install Python and Pip
1. **Download Python**:  
   - Visit the [official Python website](https://www.python.org/downloads/).  
   - Download and install **Python 3.12** (make sure to select the latest 3.12.x release).  
2. **Add Python to PATH**:  
   - During installation, ensure the **Add Python to PATH** option is checked.  
3. **Verify Installation**:  
   - Open Command Prompt and type:  
     ```bash
     py -3.12 --version
     py -3.12 -m pip --version
     ```

#### 2. Install SQLite
1. **Download SQLite**:  
   - Visit the official SQLite [download page](https://www.sqlite.org/download.html).  
   - Download the precompiled binaries for your system (e.g., `sqlite-tools-win32-x86`).  
2. **Extract SQLite**:  
   - Extract the `.zip` file into a folder (e.g., `C:\sqlite`).  
3. **Add SQLite to System PATH**:  
   - Open **Control Panel** → **System and Security** → **System** → **Advanced system settings**.  
   - Click the **Environment Variables** button.  
   - Under **System Variables**, select the `Path` variable and click **Edit**.  
   - Add the path to your SQLite folder (e.g., `C:\sqlite`).  
4. **Verify Installation**:  
   - Open Command Prompt and type:  
     ```bash
     sqlite3 --version
     ```

#### 3. Install TensorFlow + Pennylane
1. **Expand File Name Lengths** (important for TensorFlow):  
   - Windows has a default limit on file path lengths. To allow TensorFlow to install, enable long file paths:  
     - Open **Registry Editor** (`Win + R`, type `regedit`, and press Enter).  
     - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`.  
     - Find `LongPathsEnabled` and set it to `1`.  
2. **Install the packages**:  
   - Open Command Prompt and run:  
     ```bash
     py -3.12 -m pip install tensorflow pennylane
     ```  
3. **Verify Installation**:  
   - Open Python shell and type:  
     ```python
     import tensorflow as tf
     import pennylane as qml
     print(tf.__version__)
     print(qml.__version__)
     ```

---

### **Linux Installation**
#### 1. Install Python and Pip
1. **Check Python Version**:  
   - Run:  
     ```bash
     python3 --version
     ```
   - If Python 3.12 is not installed, use your package manager (example for Ubuntu/Debian):  
     ```bash
     sudo apt update
     sudo apt install python3.12 python3.12-venv python3.12-pip
     ```
2. **Verify Installation**:  
   - Run:  
     ```bash
     python3.12 --version
     python3.12 -m pip --version
     ```

#### 2. Install SQLite
1. **Check if SQLite is Installed**:  
   - Many Linux distributions include SQLite by default. Verify:  
     ```bash
     sqlite3 --version
     ```
2. **Install SQLite**:  
   - If not installed, run:  
     ```bash
     sudo apt update
     sudo apt install sqlite3
     ```

#### 3. Install TensorFlow + Pennylane
1. **Install via Pip**:  
   - Run:  
     ```bash
     python3.12 -m pip install tensorflow pennylane
     ```
2. **Verify Installation**:  
   - Open Python shell and type:  
     ```python
     import tensorflow as tf
     import pennylane as qml
     print(tf.__version__)
     print(qml.__version__)
     ```

---

### **Mac OS Installation**
*(Author's Note: Sigh, here we go...)*

#### 1. Install Python and Pip
1. **Install Homebrew** (if not already installed):  
   - Run:  
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
2. **Install Python 3.12**:  
   - Run:  
     ```bash
     brew install python@3.12
     ```
3. **Verify Installation**:  
   - Run:  
     ```bash
     python3.12 --version
     python3.12 -m pip --version
     ```

#### 2. Install SQLite
1. **Install SQLite with Homebrew**:  
   - Run:  
     ```bash
     brew install sqlite
     ```
2. **Verify Installation**:  
   - Run:  
     ```bash
     sqlite3 --version
     ```

#### 3. Install TensorFlow + Pennylane
1. **Install via Pip**:  
   - Run:  
     ```bash
     python3.12 -m pip install tensorflow pennylane
     ```
2. **Verify Installation**:  
   - Open Python shell and type:  
     ```python
     import tensorflow as tf
     import pennylane as qml
     print(tf.__version__)
     print(qml.__version__)
     ```

---

### **Install Project Dependencies**
1. **Clone the Repository**:  
   - Run:
     ```bash
     git clone https://gitlab.com/Callam7/LottoPipeline.git
     cd LottoPipeline
     ```
2. **Install Python Dependencies** (using Python 3.12):
   - Run:
     ```bash
     py -3.12 -m pip install -r requirements.txt   # Windows
     # or
     python3.12 -m pip install -r requirements.txt # Linux/macOS
     ```
3. **Verify Setup**:
   - Run the following command to ensure the project runs correctly:
     ```bash
     py -3.12 main.py   # Windows
     # or
     python3.12 main.py # Linux/macOS
     ```

You're now ready to use the Lotto Generator!

---

## Pipeline Explanation

The Lotto Pipeline is designed to simulate, analyze, and predict lottery outcomes using a series of interconnected steps. Each module contributes a specific functionality to the final output: a ticket with 12 lines of lottery numbers and corresponding Powerball numbers. Below is a chronological breakdown of the pipeline flow.

### **Core: Data Pipeline & Database**
**Module:** `pipeline.py` + `database.py`  
**Functionality**: Central data store and persistent storage.  
**How it works**  
`DataPipeline` is a simple but powerful dict wrapper with `add_data`, `get_data`, and `clear_pipeline`. Every single pipe in the entire system reads from and writes to this object — it is the single source of truth. The `database.py` file handles SQLite persistence: it creates the `draws` table (with stable `draw_id`, unique `draw_date`, comma-separated numbers, bonus 1–40, powerball 1–10) and the new `epochs` table (for deep-learning metrics). Functions like `insert_draw`, `fetch_all_draws`, `fetch_recent_draws`, and `insert_epoch_metrics` ensure safe insertion and retrieval.  
**Why it’s important**  
Without this central hub and persistent storage, data would leak between steps and there would be no way to log training history for future self-editing.

### **Step 1: Historical Data Processing**
**Module:** `steps/historical.py`  
**Functionality**: First pipe — loads and cleans draws.  
**How it works**  
It receives `{"past_results": [...]}` from the database, filters out any draw where `powerball` is not an integer between 1 and 10, and stores the clean list under the key `"historical_data"`.  
**Why it’s important**  
This is the foundation — every later calculation (frequencies, decay, fusion, quantum training, etc.) depends on this validated data.

### **Step 2: Frequency Analysis**
**Module:** `steps/frequency.py`  
**Functionality**: Global occurrence probabilities.  
**How it works**  
It loops through all historical draws, collects every valid main number (1–40) and Powerball (handles both int and list formats), uses `np.bincount` to count occurrences, and normalises to probabilities. It stores three vectors: main (40,), Powerball (10,), and combined (50,). Invalid numbers are logged as warnings.  
**Why it’s important**  
This is the baseline occurrence signal that decay, Bayesian fusion, Monte Carlo, and deep learning all build upon.

### **Step 3: Decay Factors Calculation**
**Module:** `steps/decay.py`  
**Functionality**: Recency weighting.  
**How it works**  
It parses every draw date (robust `_safe_parse_date` that handles strings and datetime objects), finds the most recent date, calculates weeks since then for each draw, applies exponential decay `decay_factor = 0.98 ** weeks_passed`, accumulates weighted counts separately for main and Powerball, normalises each group, and concatenates to a (50,) vector stored as `"decay_factors"`.  
**Why it’s important**  
Recent draws carry more influence — this adds temporal bias without discarding older data.

### **Step 4: Bayesian Fusion with Mechanics**
**Module:** `steps/bayesian_fusion_with_mechanics.py`  
**Functionality**: Mechanics estimation + fusion.  
**How it works**  
It computes a Dirichlet posterior mean for main numbers (smoothed counts + alpha), runs a chi-square test against uniform, and collapses to uniform if the signal is not statistically significant. It then fuses frequency + decay + mechanics in log-space: `posterior = exp(w_f·log(freq) + w_d·log(decay) + w_m·log(mechanics))`, normalises, and stores both the true probability vector and a max-normalised version for deep learning.  
**Why it’s important**  
This produces the coherent prior that clustering, Monte Carlo, and deep learning use.

### **Step 5: Clustering and Correlation**
**Module:** `steps/clustering.py`  
**Functionality**: K-Means on fusion probabilities.  
**How it works**  
It takes the Bayesian fusion vector, scales each group (main/PB) with MinMaxScaler, dynamically reduces cluster count if variance is tiny, runs KMeans, and stores cluster IDs and centroid strengths per number (tiled later into deep learning).  
**Why it’s important**  
It reveals hidden groupings that raw frequencies miss.

### **Step 6: Monte Carlo Simulations**
**Module:** `steps/monte_carlo.py`  
**Functionality**: Sampling-based probability estimation.  
**How it works**  
It dynamically sets simulation count based on data size, adjusts fusion probabilities by cluster centroids, runs main-number simulations (no replacement) and Powerball simulations, counts outcomes, normalises, and stores the (50,) vector as `"monte_carlo"`.  
**Why it’s important**  
It adds robustness through repeated random sampling.

### **Step 7: Sequential / Temporal Features**
**Module:** `steps/redundancy.py`  
**Functionality**: Recency + gap statistics.  
**How it works**  
It calculates recency (1 – age/total_draws) and full unbiased gaps (initial + internal + final), normalises each by their standard deviation so they contribute equally, averages them, multiplies by centroids, and applies strict min-max normalisation to [0,1]. The result is stored as `"redundancy"`.  
**Why it’s important**  
It adds temporal context that Markov and entropy later weight against.

### **Step 8: First-Order Markov Chain**
**Module:** `steps/markov.py`  
**Functionality**: Inter-draw cluster transitions.  
**How it works**  
It converts each draw into one representative cluster (mode), builds a proper transition matrix between consecutive draws, scores each number by the probability of transitioning from the last draw’s cluster to its own, multiplies by redundancy weights, and normalises. The result is stored as `"markov_features"`.  
**Why it’s important**  
It models sequence dependencies across draws.

### **Step 9: Shannon Entropy Features**
**Module:** `steps/entropy.py`  
**Functionality**: Per-symbol unpredictability.  
**How it works**  
It takes the Bayesian fusion probabilities, computes the Shannon entropy contribution `-p_i * log2(p_i)` for each of the 50 numbers (with safe clipping), normalises the vector to sum to 1, and stores it as `"entropy_features"`.  
**Why it’s important**  
It highlights which numbers are most “random” versus structured.

### **Step 10: Quantum Encoder Training**
**Module:** `config/quantum_features.py`  
**Functionality**: SPSA-trained variational quantum circuit.  
**How it works**  
A deterministic classical projection mixes the input features into 12 dimensions using the fixed matrix `M[q,j] = sin((q+1)(j+1)) + 0.5*cos((q+1)(j+1))`. These are standardised, clipped, and mapped to angles in [-π, π]. The circuit applies Hadamard gates, RY rotations, and 3 layers of StronglyEntanglingLayers. Z-expectations are measured and augmented with mean, std, L1, and L2². SPSA updates the circuit weights θ using the two-point gradient estimate `g_hat = ((loss_plus - loss_minus) / (2*c_t)) * delta` with the standard SPSA schedules. The trained weights are stored globally.  
**Why it’s important**  
It learns a supervised nonlinear quantum feature map that classical networks cannot replicate exactly.

### **Step 11: Quantum Features & Kernel**
**Module:** `config/quantum_features.py` + `config/quantum_kernels.py`  
**Functionality**: Quantum matrix + fidelity kernels.  
**How it works**  
`compute_quantum_matrix` runs the circuit on every row to produce 12 Z-values plus 4 summary statistics (16-dim total). `build_quantum_kernel_features` encodes prototype rows into statevectors, computes the fidelity kernel `K[i,j] = |⟨ψ(x_i)|ψ(proto_j)⟩|²`, then applies Mercer-safe diagonal normalisation `K' = K · D^(-1/2) · D^(-1/2)` to guarantee positive semi-definiteness and numerical stability. Prototypes are cached and reset after encoder training.  
**Why it’s important**  
It adds entangled representations and similarity-based features that are fused directly into deep learning.

### **Step 12: Deep Learning Fusion Model**
**Module:** `steps/deep_learning.py`  
**Functionality**: Final nonlinear synthesis.  
**How it works**  
It builds a causal prefix-frequency matrix, reweights it by all upstream signals, tiles centroids/clusters/quantum predictions, adds the quantum matrix and kernel features, trains a feed-forward network with weighted BCE (positives-only weighting, no label smoothing), mild Gaussian augmentation, and val_auc early stopping. The final (50,) vector is stored as `"deep_learning_predictions"`.  
**Why it’s important**  
This is the core predictive model that turns every prior signal into the final probabilities.

### **Final Step: Ticket Generation**
**Module:** `steps/generate_ticket.py`  
**Functionality**: Creates playable 12-line ticket.  
**How it works**  
It starts with deep learning probabilities, applies cross-line decay penalties after each accepted line, and uses rejection sampling (up to 250 tries) to enforce hard constraints: no line may share >2 main numbers with any previous line, and the same Powerball may appear at most twice. It saves the ticket to `current_ticket.json`.  
**Why it’s important**  
It turns raw probabilities into diverse, usable tickets.

### **Additional Components**
**Module:** `data_io.py`  
**Functionality**: JSON save/load for tickets.  
**How it works**  
`save_current_ticket` validates each line (must have “line” and “powerball” keys, all integers), normalises, and writes with indentation. `load_current_ticket` is defensive against missing files or corrupt JSON.  
**Why it’s important**  
It provides clean persistence for the menu and future use.

**Module:** `config/logs.py`  
**Functionality**: EpochLogger callback.  
**How it works**  
It inherits from `tf.keras.callbacks.Callback`, runs after every epoch, extracts all metrics (handling both legacy and new names), and calls `insert_epoch_metrics` to store them in the `epochs` table grouped by `run_date`. A small sleep prevents DB contention.  
**Why it’s important**  
It enables the future self-editing model by logging full training history.

---

## Future Additions

My next focus will be on implementing thorough pytesting for each pipeline step to ensure correctness, stability, and consistent outputs. Automated tests will validate input handling, edge cases, and integration between pipes.

Following testing, I will plan on building a comprehensive logging system that captures detailed metrics and stats for every epoch and pipeline execution. These logs will be stored systematically for auditing, debugging, and performance tracking over time.

Finally, I plan to integrate automation that leverages a self-editing model. This model will dynamically adapt the pipeline by analyzing logs and test results, enabling continuous improvement and fine-tuning of prediction accuracy and processing efficiency without manual intervention. Essentially, a algorithm that evolves its own code based on performance. The theory is that as it evolves and changes, it will have improving performance that will eventually reach pin-point accurate prediction capabilities.

---
