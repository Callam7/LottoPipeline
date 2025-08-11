# Lotto Generator

**Author's Note**  
This project will not predict lottery numbers with perfect accuracy. It’s a proof of concept to test ideas, build a scalable pipeline, and see how far different methods can push results against noisy, pseudo-random data.

It started as a challenge to the claim that Lotto is “just random.” While draws are unpredictable, they follow fixed rules — main numbers 1–40, Powerball 1–10 — which makes them pseudo-random. That means patterns can be measured and methods applied to guide number selection.

The project has since shifted toward laying the groundwork for a self-editing adaptive model. The idea is for the system to train on historical data, evaluate its own performance, and adjust over time. This hasn’t been reached yet, but the current pipeline is built to allow it.

---

## Purpose

The **Lotto Generator** is a Python application that analyzes historical lottery data and generates tickets based on multiple statistical and machine learning methods.

It uses a pipeline of “pipes” — statistical analysis, clustering, Monte Carlo simulation, Markov chains, Shannon entropy, and deep learning. Each pipe produces features from the historical draws, which are combined and used to train the deep learning model.

The long-term goal is for the pipeline to update and re-train itself automatically as new draws come in, with the aim of improving accuracy while avoiding overfitting. For now, Lotto is just the sandbox for testing these methods.

---

## Table of Contents
1. [Introduction](#lotto-generator)  
2. [Purpose](#purpose)   
3. [Installation Instructions](#installation-instructions)  
4. [Pipeline Explanation](#pipeline-explanation)  
5. [Future Additions](#future-additions)  

---

## Installation Instructions

To set up the project locally, follow these step-by-step instructions for your operating system. The process includes installing the required Python version, necessary packages, and configuring specific dependencies for TensorFlow and SQLite.

### Prerequisites
1. **Python**: Ensure you have Python 3.8 or higher installed.  
2. **Pip**: The Python package manager (comes with Python 3.4+).  
3. **SQLite**: SQLite must be installed and added to your system's PATH.  
4. **TensorFlow**: Proper installation steps for TensorFlow depending on the OS.

---

### **Windows Installation**
#### 1. Install Python and Pip
1. **Download Python**:  
   - Visit the [official Python website](https://www.python.org/downloads/).  
   - Download and install Python (3.8 or higher).  
2. **Add Python to PATH**:  
   - During installation, ensure the **Add Python to PATH** option is checked.  
3. **Verify Installation**:  
   - Open Command Prompt and type:  
     ```bash
     python --version
     pip --version
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
   - You should see the SQLite version number.

#### 3. Install TensorFlow
1. **Expand File Name Lengths**:  
   - Windows has a default limit on file path lengths. To allow TensorFlow to install, enable long file paths:  
     - Open **Registry Editor** (`Win + R`, type `regedit`, and press Enter).  
     - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`.  
     - Find `LongPathsEnabled` and set it to `1`.  
2. **Install TensorFlow**:  
   - Open Command Prompt and run:  
     ```bash
     pip install tensorflow
     ```  
3. **Verify TensorFlow Installation**:  
   - Open Python shell and type:  
     ```python
     import tensorflow as tf
     print(tf.__version__)
     ```
   - The TensorFlow version should be displayed.

---

### **Linux Installation**
#### 1. Install Python and Pip
1. **Check Python Version**:  
   - Run:  
     ```bash
     python3 --version
     ```
   - If Python is not installed, use your package manager:  
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip
     ```  
2. **Verify Installation**:  
   - Run:  
     ```bash
     python3 --version
     pip3 --version
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

#### 3. Install TensorFlow
1. **Install TensorFlow via Pip**:  
   - Run:  
     ```bash
     pip3 install tensorflow
     ```
2. **Verify Installation**:  
   - Open Python shell and type:
     ```python
     import tensorflow as tf
     print(tf.__version__)
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
2. **Install Python with Homebrew**:  
   - Run:  
     ```bash
     brew install python
     ```
3. **Verify Installation**:  
   - Run:  
     ```bash
     python3 --version
     pip3 --version
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

#### 3. Install TensorFlow
1. **Install TensorFlow via Pip**:  
   - Run:  
     ```bash
     pip3 install tensorflow
     ```
2. **Verify Installation**:  
   - Open Python shell and type:
     ```python
     import tensorflow as tf
     print(tf.__version__)
     ```

---

### **Install Project Dependencies**
1. **Clone the Repository**:  
   - Run:
     ```bash
     git clone https://github.com/your-repo/LottoPipeline.git
     cd LottoPipeline
     ```
2. **Install Python Dependencies**:
   - Run:
     ```bash
     pip install -r requirements.txt
     ```
3. **Verify Setup**:
   - Run the following command to ensure the project runs correctly:
     ```bash
     python main.py
     ```

You're now ready to use the Lotto Generator!

---

## Pipeline Explanation

The Lotto Predictor pipeline is designed to simulate, analyze, and predict lottery outcomes using a series of interconnected steps. Each module contributes a specific functionality to the final output: a ticket with 12 lines of lottery numbers and corresponding Powerball numbers. Below is a chronological breakdown of the pipeline flow:

### **Step 1: Historical Data Processing**
**Module:** `steps/historical.py`  
- **Functionality**: This is the first stage of the pipeline. It loads, validates, and filters historical lottery draws before any other processing takes place. The idea is to ensure every downstream method works only with clean, valid data.

- **How it works**  
 - Retrieves past draw results from the input or connected database  
 - Filters out any draw with an invalid Powerball (must be 1–10)  
 - Passes only valid draws forward through the pipeline  

- **Why it’s important:**  
 Garbage in means garbage out. By validating historical draws at the start, the rest of the pipeline is trained and tested on trustworthy data. This step directly improves the reliability of all downstream statistical, clustering, and machine learning processes.

### **Step 2: Frequency Analysis**
**Module:** `steps/frequency.py`  
- **Functionality**: Calculates how often each main number (1–40) has been drawn historically, then converts those counts into a normalized probability distribution. This ensures later stages of the pipeline can work with weighted probabilities instead of treating all numbers equally.
 
- **How it works**:  
  - Loads historical draw data from the pipeline  
  - Filters out invalid numbers outside the 1–40 range  
  - Counts how many times each number appears in valid draws  
  - Normalizes the counts so the total probability equals 1  
  - Stores the probability distribution in the pipeline for later steps

**Why it’s important**:
- This step lets downstream models prioritize numbers that appear more often in past draws, without overfitting. If no valid data exists, a uniform distribution is used to keep the pipeline functional.

### **Step 3: Decay Factors Calculation**
**Module:** `steps/decay.py`  
- **Functionality**: Applies a decay factor to historical lottery results so that recent draws have more influence than older draws. This is based on the assumption that more recent outcomes may better reflect current draw trends.
  
- **How it works**:  
   - Retrieves historical draw data from the pipeline  
   - Sorts draws chronologically by their draw date  
   - For each draw, calculates the time difference from the previous draw in weeks  
   - Applies a decay rate (default 0.98) raised to the number of weeks passed  
   - Adds decay-weighted counts for both main numbers (1–40) and Powerball numbers (1–10)  
   - Normalizes the counts into probability distributions so the total probability equals 1  
   - Stores the decay-weighted probabilities in the pipeline under `decay_factors`

**Why it’s important**:
- This step allows the model to focus more on recent patterns without discarding older results entirely. By weighting data based on recency, it captures potential short-term trends while still incorporating long-term historical context.

### **Step 4: Clustering and Correlation**
**Module:** `steps/clustering.py`  
- **Functionality**: Groups lottery numbers into clusters based on their historical draw frequencies. This helps reveal underlying patterns or tendencies that may not be obvious from raw counts alone.
 
- **How it works**:  
   - Retrieves normalized frequency data for both main numbers (1–40) and Powerball numbers (1–10) from the pipeline  
   - Scales the data to a [0, 1] range using MinMaxScaler for consistent clustering behavior  
   - Applies K-Means clustering separately to main numbers and Powerball numbers  
   - Dynamically reduces cluster count if the variance in frequency data is too low to form distinct groups  
   - Extracts cluster labels and centroids for each group  
   - Stores results in the pipeline:
  	- `clusters` and `centroids` for main numbers  
  	- `powerball_clusters` and `powerball_centroids` for Powerball numbers  
  	- `number_to_cluster` mapping for quick reference

- **Why it’s important**:
- By grouping numbers with similar draw patterns, the model can identify trends that extend beyond simple frequency counts. This provides a richer feature set for downstream predictive modeling and can be combined with decay-weighted data, entropy measures, or other features to improve prediction stability.

### **Step 5: Monte Carlo Simulations**
**Module:** `steps/monte_carlo.py`  
- **Functionality**: Generates multiple simulated lottery outcomes based on probability distributions from earlier pipeline steps, allowing the model to estimate likely future draws.
  
- **How it works**:  
   - Retrieves probability distributions for main numbers (1–40) and Powerball numbers (1–10) from the pipeline  
   - Runs a set number of simulations (e.g., 10,000)  
   - In each simulation:  
     - Randomly selects 6 unique main numbers based on their probabilities  
     - Randomly selects 1 Powerball number based on its probability distribution  
   - Tracks how often each number appears across all simulations  
   - Normalizes the counts into probability distributions  
   - Stores results in the pipeline under `monte_carlo_results`

**Why it’s important**:  
- Monte Carlo simulation uses repeated random sampling to account for probability-based uncertainty. It helps the model identify numbers that consistently appear across thousands of trials, improving prediction reliability.

### Step 5: Sequential / Temporal Feature Generation (Redundancy)
**Module:** `steps/redundancy.py`
- **Functionality**: Extracts temporal features from historical lottery draws, capturing how recently each number was drawn and the average gaps between its appearances. These features provide context on number recurrence patterns to improve prediction models.

**How it works**  
  - Retrieves historical draw data from the pipeline  
  - Calculates recency for each number (1–40): how many draws ago it last appeared, normalized between 0 (recent) and 1 (old)  
  - Calculates average gap length between appearances for each number, normalized by total draws  
  - Combines recency and gap features into a single 40-element feature vector  
  - Stores the combined features in the pipeline under `redundancy`

**Why it’s important**:  
- Including sequential and temporal features helps the model understand patterns beyond simple frequency, like how often numbers repeat or cluster over time. This enriches the predictive signal for deep learning by adding sequential context.


### **Step 6: First-Order Markov Chain Transition Features**
**Module:** `steps/markov.py`  
- **Functionality**: Generates features representing the probabilities of transitioning from one cluster of numbers to another based on historical draw sequences. This captures temporal dynamics of cluster behavior to enhance prediction.
  
- **How it works**:  
  - Retrieves historical draw data, cluster assignments for numbers, and redundancy features from the pipeline  
  - Constructs a sequence of average cluster IDs for each draw by averaging the clusters of drawn numbers  
  - Builds a transition matrix showing probabilities of moving from one cluster state to another (first-order Markov chain)  
  - Scores each number based on the average transition probabilities of its cluster  
  - Optionally weights scores by redundancy (temporal features)  
  - Normalizes scores into a probability distribution and stores them in the pipeline under `markov_features`

**Why it’s important**:
- Markov transition features encode the temporal dependencies between clusters of numbers, allowing the model to leverage patterns of cluster transitions over time, which static frequency counts alone cannot capture.

### **Step 7: Shannon Entropy Features**
**Module:** `steps/entropy.py`  
- **Functionality**: Calculates the Shannon entropy rate of the lottery number process using Markov chain transitions between clusters. Entropy measures the unpredictability of number sequences, helping identify how much randomness or structure exists in cluster transitions.
  
- **How it works**:  
   - Retrieves clusters, number-to-cluster mappings, centroids, redundancy, Markov scores, and historical data from the pipeline  
   - Builds a sequence of clusters representing each historical draw’s average cluster  
   - Constructs a Markov transition matrix between clusters based on this sequence  
   - Computes the stationary distribution (long-term probabilities) of the Markov chain  
   - Calculates the entropy rate (overall unpredictability) and entropy per cluster state  
   - Assigns entropy scores to each number according to its cluster  
   - Applies redundancy weighting to emphasize recently active numbers  
   - Normalizes the entropy scores into a probability distribution  
   - Stores the entropy features in the pipeline under `entropy_features`

**Why it’s important**:  
- Entropy quantifies the amount of uncertainty in the lottery number sequences. Including entropy-based features allows the model to capture and leverage the complexity and randomness inherent in number transitions, improving predictive insights.

### **Step 8: Deep Learning Model (Tensorflow/Binary Crossentropy**
**Module:** `steps/deep_learning.py`
- **Functionality**: Attempts to predict probabilities for the 40 main lottery numbers using a deep learning model. It integrates diverse features from earlier steps—historical data, decay-weighted frequencies, Monte Carlo simulations, clustering, redundancy, Markov transition probabilities, and entropy—into a single predictive framework.

**How it works**  
  - Retrieves necessary input data from the pipeline, including historical draws, decay factors, Monte Carlo results, clusters, centroids, redundancy features, Markov features, and entropy features.  
  - Normalizes all input features individually to ensure consistent scale.  
  - Combines normalized features along with cluster centroid information into a feature matrix (shape: 40 numbers × multiple features).  
  - Constructs binary multi-label targets from historical draws, applying label smoothing to reduce overconfidence.  
  - Calculates class weights to counter class imbalance (numbers drawn less frequently get higher weights).  
  - Defines a custom weighted binary crossentropy loss function incorporating class weights.  
  - Performs data augmentation by injecting Gaussian noise into the input features to improve generalization.  
  - Sets dynamic training parameters (number of epochs, etc.) based on dataset size.  
  - Builds a neural network model with two hidden layers (128 and 64 units), ReLU activations, batch normalization, dropout for regularization, and sigmoid output for independent probabilities per number.  
  - Compiles the model with Adam optimizer and the custom loss function, tracking binary accuracy, AUC, and MAE metrics.  
  - Trains the model on augmented data using early stopping and learning rate reduction callbacks to optimize training.  
  - Predicts the probability for each number independently on the original feature set and averages predictions.  
  - Stores the resulting normalized probability distribution of number predictions in the pipeline under `deep_learning_predictions`.

**Why it’s important**:  
- This step synthesizes all prior analytical features into a learned nonlinear model that can capture complex interactions and dependencies among lottery numbers. It outputs a refined probability distribution reflecting the model’s best estimate of each number’s likelihood, forming the core predictive signal for ticket generation.


### **Final Step: Ticket Generation**
**Module:** `steps/generate_ticket.py`
- **Functionality**: Generates a full set of lottery ticket lines using the final deep learning predictions. This step consolidates all prior analysis into concrete ticket lines for play.

**How it works**  
  - Retrieves the deep learning probability distribution for the 40 main numbers from the pipeline.  
  - Applies a safety fallback to a uniform distribution if predictions are missing.  
  - Normalizes probabilities to ensure they sum to 1 and clips to avoid zeros.  
  - Uses a diversity penalty mechanism to reduce repetition across the 12 generated lines, by lowering the probabilities of numbers already selected in prior lines.  
  - For each line, randomly selects 6 unique main numbers based on the adjusted probabilities, ensuring no duplicate lines are created.  
  - Selects the Powerball number weighted by decay-weighted historical frequencies if available; otherwise uses a uniform distribution.  
  - Updates frequency penalties after each line to promote diversity in future lines.  
  - Saves the final set of 12 ticket lines (main numbers + Powerball) to a JSON file for downstream use.

**Why it’s important**:  
- This final step transforms the probabilistic predictions into actual ticket selections while maintaining diversity and respecting the nuanced influences learned throughout the pipeline. It operationalizes the model output into actionable lottery tickets.

---

### 5. Future Additions

My next focus will be on implementing thorough pytesting for each pipeline step to ensure correctness, stability, and consistent outputs. Automated tests will validate input handling, edge cases, and integration between pipes.

Following testing, I will plan on building a comprehensive logging system that captures detailed metrics and stats for every epoch and pipeline execution. These logs will be stored systematically for auditing, debugging, and performance tracking over time.

Finally, we plan to integrate automation that leverages a self-editing model. This model will dynamically adapt the pipeline by analyzing logs and test results, enabling continuous improvement and fine-tuning of prediction accuracy and processing efficiency without manual intervention.

---



