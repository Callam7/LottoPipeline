# Lotto Generator

**Author's Note**:  
There is no guarantee that this project will predict lottery numbers with pinpoint accuracy. This is a proof of concept, designed to showcase my ability to develop a scalable pipeline and explore the potential of data-driven methods. 

I created this project because I found it a bit annoying when Lotto advertised that "there is no method to playing Lotto; it's just random." While it's true that lottery draws are highly unpredictable, this statement isn't entirely accurate. If we consider the finite set of rules they’ve put in place—numbers between 1-40 for the main draw and 1-10 for the Powerball—we can see that it’s not truly random but *pseudo-random*. With a sophisticated approach, patterns can absolutely be analyzed, and methods can be applied to improve decision-making. This project was designed to challenge that misconception and to explore how far we can push those boundaries using modern computational techniques.

---

## Purpose
The **Lotto Generator** is an advanced Python-based application designed to analyze historical lottery draw data and generate optimized lottery tickets with a higher likelihood of success.

By combining statistical techniques, clustering algorithms, Monte Carlo simulations, and machine learning models, the pipeline identifies patterns in historical draws and uses this information to predict future trends. This allows for data-driven ticket generation, maximizing diversity and leveraging past performance to improve probabilities.

---

## Key Functionalities
- **Data Analysis**: Processes historical lottery data to uncover frequency distributions and decay factors.  
- **Probability Estimation**: Simulates thousands of potential draws using Monte Carlo methods and predictive models.  
- **Machine Learning**: Utilizes a deep learning neural network to refine predictions based on past outcomes.  
- **Ticket Optimization**: Employs a diversity mechanism to generate distinct and balanced tickets, reducing redundancy.  

This project bridges the gap between statistical analysis and predictive modeling, making it a powerful tool for anyone looking to optimize their lottery strategies.

---

## Table of Contents
1. [Introduction](#lotto-generator)  
2. [Purpose](#purpose)  
3. [Key Functionalities](#key-functionalities)  
4. [Features](#features)  
5. [Installation Instructions](#installation-instructions)  
6. [Pipeline Explanation](#pipeline-explanation)  
7. [Project Structure](#project-structure)  

---

## Features

1. **Dynamic Pipeline for Lottery Data Analysis**  
   - A scalable and modular pipeline designed to process, analyze, and predict lottery outcomes.  
   - Integrates multiple stages such as historical data processing, frequency analysis, clustering, decay factor application, and deep learning.  
   - Each step is independent, allowing for easy updates or replacements without breaking the pipeline.

2. **Advanced Statistical and Mathematical Analysis**  
   - Incorporates sophisticated statistical models to analyze lottery trends and probabilities.  
   - Uses clustering (via K-Means) to identify hidden patterns in historical draw data.  
   - Applies decay factors to give recent draws more weight, simulating a realistic and dynamic approach to probability adjustments.

3. **Monte Carlo Simulation**  
   - Generates thousands of random lottery combinations based on weighted probabilities.  
   - Provides a robust probability distribution for the 40 main numbers and Powerball.

4. **Deep Learning for Prediction**  
   - Trains a neural network to predict the likelihood of each lottery number appearing in future draws.  
   - Combines decay factors, clustering information, and Monte Carlo outputs for enhanced accuracy.  
   - The AI model adapts dynamically to the size and scope of historical data, using a sophisticated training regimen.

5. **Ticket Generation with Diversity Penalty**  
   - Generates 12 unique lottery lines with a penalty-based mechanism to ensure number diversity.  
   - Balances predictions from the neural network and decay factors for optimal number selection.  
   - Prevents repetitive combinations to increase the coverage of possible outcomes.

6. **Comprehensive Database Integration**  
   - Stores historical draw data and user-generated tickets in a robust SQLite database.  
   - Ensures chronological ordering and consistency of draw records.  
   - Fetches recent or specific historical draws for analysis.

7. **User-Friendly Console Menu**  
   - Provides an interactive menu for:
     - Viewing current tickets.
     - Listing recent draws.
     - Adding new draws and generating tickets.
     - Viewing number frequency statistics.  
   - Guides users step-by-step through all available functionalities.

8. **Scalability and Extensibility**  
   - The modular design allows for future improvements, such as additional AI models or new statistical techniques.  
   - Can be adapted for other applications involving probabilistic modeling and prediction.

9. **Proof of Concept for Pseudo-Random Systems**  
   - Challenges the misconception of true randomness in lotteries by demonstrating the use of structured methods to analyze and predict outcomes.  
   - Validates the hypothesis that lottery results, being a finite set within defined rules, follow pseudo-random patterns.

10. **Data Visualization (Planned Feature)**  
   - Planned implementation of visualizing trends and predictions to enhance user insights.  
   - Graphs and heatmaps will display the frequency distribution of numbers and prediction accuracies.

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
- **Functionality**: Processes historical lottery draw data to ensure only valid draws are included in the pipeline.  
- **Details**:  
  - Filters out draws with invalid Powerball numbers or missing data.  
  - Stores the cleaned historical data in the pipeline for downstream use.

### **Step 2: Frequency Analysis**
**Module:** `steps/frequency.py`  
- **Functionality**: Analyzes the frequency of each lottery number based on historical draws.  
- **Details**:  
  - Calculates how often each number (1-40) appears in historical draws.  
  - Produces a normalized probability distribution to emphasize frequently drawn numbers.  
  - Adds this distribution to the pipeline for later processing.

### **Step 3: Decay Factors Calculation**
**Module:** `steps/decay.py`  
- **Functionality**: Applies a time-based decay to historical data, giving more weight to recent draws.  
- **Details**:  
  - Adjusts the frequency of numbers and Powerball values based on their recency.  
  - Creates a probability distribution where recent numbers are weighted more heavily than older ones.  
  - Adds the decay-adjusted frequency data to the pipeline.

### **Step 4: Clustering and Correlation**
**Module:** `steps/clustering.py`  
- **Functionality**: Groups numbers into clusters based on their frequency distribution to identify patterns.  
- **Details**:  
  - Uses K-Means clustering to group numbers into clusters.  
  - Calculates centroids for each cluster to emphasize or de-emphasize certain groups.  
  - Adds cluster labels and centroids to the pipeline for probability adjustment.

### **Step 5: Monte Carlo Simulations**
**Module:** `steps/monte_carlo.py`  
- **Functionality**: Simulates lottery draws using adjusted probabilities to refine number predictions.  
- **Details**:  
  - Combines decay factors, cluster centroids, and frequency data to adjust probabilities.  
  - Runs Monte Carlo simulations to generate a large dataset of simulated draws.  
  - Analyzes these simulations to create a refined probability distribution.  
  - Adds this enhanced distribution to the pipeline.

### **Step 6: Deep Learning Predictions**
**Module:** `steps/deep_learning.py`  
- **Functionality**: Trains a neural network to predict lottery number probabilities based on all previous data.  
- **Details**:  
  - Combines Monte Carlo data, decay factors, and clustering outputs into feature vectors.  
  - Trains a neural network with historical data using data augmentation to improve generalization.  
  - Outputs a refined probability distribution for the lottery numbers.  
  - Adds the deep learning predictions to the pipeline.

### **Step 7: Ticket Generation**
**Module:** `steps/generate_ticket.py`  
- **Functionality**: Combines all refined probabilities to generate 12 diverse and optimized ticket lines.  
- **Details**:  
  - Uses a weighted combination of deep learning predictions and decay-adjusted probabilities.  
  - Implements a penalty-based diversity mechanism to ensure no repeated ticket lines.  
  - Selects Powerball numbers based on decay-adjusted probabilities.  
  - Saves the generated ticket lines in a JSON file for user reference.

### **Final Output: A Complete Ticket**
The pipeline produces a ticket with 12 unique lines of 6 main numbers (1-40) and 1 Powerball number (1-10). Each number is carefully calculated using historical data, frequency analysis, decay factors, clustering, Monte Carlo simulations, and deep learning.

This multi-step approach ensures the final ticket is not entirely random but informed by statistical patterns and refined probabilities.

---

## Project Structure

The project is organized into clearly defined files and folders to ensure modularity, readability, and ease of maintenance. Below is an overview of the structure and the purpose of each major component:

### Root Directory
- **`current_ticket.json`**: A dynamically updated file that stores the most recently generated lottery ticket in JSON format.  
- **`database.py`**: Handles all SQLite database interactions, including initializing the database, inserting draws, and fetching historical data.  
- **`data_io.py`**: Manages the input/output of generated ticket data, ensuring the integrity of saved and loaded tickets.  
- **`main.py`**: The entry point of the project, providing an interactive console menu for managing lottery draws, generating tickets, and viewing statistics.  
- **`pipeline.py`**: Defines the `DataPipeline` class, enabling data sharing across different processing steps, and implements utility functions for dynamic parameter generation.

---

### `/steps`
This folder contains modular scripts responsible for each stage of the pipeline. Each script processes data or generates probabilities for lottery predictions.

- **`clustering.py`**: Performs K-Means clustering on number frequency data to uncover patterns and correlations.  
- **`decay.py`**: Calculates decay factors to give more weight to recent draws and adjusts probabilities dynamically.  
- **`deep_learning.py`**: Trains a neural network to predict the likelihood of each lottery number appearing in future draws.  
- **`frequency.py`**: Analyzes the frequency of lottery numbers from historical data and generates a normalized probability distribution.  
- **`generate_ticket.py`**: Combines neural network predictions and decay factors to generate diverse and weighted lottery tickets.  
- **`historical.py`**: Processes historical draw data, validates it, and integrates it into the pipeline.  
- **`monte_carlo.py`**: Runs Monte Carlo simulations to create probabilistic models for lottery number prediction.

---

### Project Overview

| **File/Folder**          | **Purpose**                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `current_ticket.json`    | Stores the most recently generated lottery ticket in JSON format.                             |
| `database.py`            | Manages the SQLite database for storing and querying draw data.                               |
| `data_io.py`             | Handles saving and loading ticket data to/from `current_ticket.json`.                         |
| `main.py`                | Entry point of the project, providing a user-friendly interface for running the pipeline.     |
| `pipeline.py`            | Implements the data pipeline to coordinate and share data across different steps.            |
| `/steps/clustering.py`   | Clusters number frequencies using K-Means to identify patterns.                               |
| `/steps/decay.py`        | Adjusts probabilities dynamically using decay factors to emphasize recent draws.             |
| `/steps/deep_learning.py`| Uses a neural network to predict the probability of each lottery number appearing.            |
| `/steps/frequency.py`    | Computes the normalized frequency of numbers from historical data.                            |
| `/steps/generate_ticket.py` | Generates 12 diverse ticket lines by combining decay factors and neural network predictions. |
| `/steps/historical.py`   | Processes and validates historical draw data for use in the pipeline.                         |
| `/steps/monte_carlo.py`  | Runs Monte Carlo simulations to create probabilistic models for number selection.             |

