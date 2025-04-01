# FL Simulator

FL Simulator is a federated learning simulation platform designed for both academic research and industrial applications. By leveraging [Ray](https://www.ray.io) for parallel training and a flexible Python-based backend alongside a [Streamlit](https://streamlit.io) frontend, the platform enables rapid experimentation with various federated learning scenarios, attack schemes, and defense mechanisms.

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Using Docker](#using-docker)
- [Configuration](#configuration)
- [Attack & Defense Schemes](#attack--defense-schemes)
- [Logging & Querying](#logging--querying)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- **Scalable Federated Learning:**  
  Fully distributed simulations leveraging Ray for parallel client training to reduce run-times.

- **Modular Design:**  
  Easily extend the platform with custom attack methods, defense mechanisms, datasets, and models.

- **Wide Range of Attacks:**  
  A variety of adversarial attack schemes have been implemented, including:
  - Label Flip Attacks (https://arxiv.org/abs/1206.6389)
  - Min-Max & Min-Sum Attacks (https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/)
  - Krum & Trim Attacks (https://www.usenix.org/conference/usenixsecurity20/presentation/fang)
  - Gaussian Noise Attacks
  - Sign Flip Attacks (https://ieeexplore.ieee.org/document/10725463)
  - No_Attack : used to run a clean training process.

- **Aggregation Strategies:**  
  A variety of aggregation strategies have been implemented to mitigate adversarial attacks, including:
  - FEDAVG (https://proceedings.mlr.press/v54/mcmahan17a.html)
  - KRUM (https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
  - MEDIAN (https://proceedings.mlr.press/v80/yin18a.html)
  - TRIM_MEAN (https://proceedings.mlr.press/v80/yin18a.html)
  - KeTS (https://arxiv.org/abs/2501.06729)
  - FLTRUST (https://arxiv.org/abs/2012.13995)
  - Custom variants for research purposes

- **Built-in Differential Privacy:**  
  Integrates [Opacus](https://opacus.ai/) to enable differentially private training, ensuring privacy-preserving simulations.

- **Interactive Frontend:**  
  Uses Streamlit for a user-friendly interface to set simulation parameters, monitor training, and query past experiment results.

- **Comprehensive Logging & Database Querying:**  
  Logs simulation details to the console and stores results in a SQL database for historical tracking and further analysis.

## Project Structure

- **backend/** – Contains the Python backend code that implements the simulation logic, configuration handling, database integration, and API endpoints.
  - *utils/* – Utility modules for configuration, logging, and database management.
  - *app/* – Flask application with routes for simulation, configuration, and query execution.
  - *server.py* – Contains the core logic for managing the training process, implementing attacks, and handling defense mechanisms.
  - *client.py* - Houses the primary functionality for executing client-side training.
  - *dataset* - CManages dataset loading. It selects the partitioner and loads both client-specific datasets and a centralized test set. It currently supports:   – Tabular datasets: kdd-cup-1999, adult-census-income, covtype.   – Image datasets: MNIST, Fashion_MNIST, cifar10.
- **streamlit_app.py** – The Streamlit-based frontend for interactive simulation configuration and monitoring.
- **docker-compose.yml** – Docker Compose configuration for orchestrating the services.
- **Dockerfile-streamlit** – Dockerfile to build the Streamlit container.
- **requirements.txt** – List of required Python packages.
- **README.md** – This file.

## Installation & Setup

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/) (optional, if you want to run with Docker)
- pip

### Local Setup

1. **Clone the Repository:**

   ```sh
   git clone <repository_url>
   cd FL_Simulator
   ```

2. **Create & Activate Virtual Environment:**

   - On Linux/macOS:
     ```sh
     python -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running Locally

1. **Start the Backend:**

   In one terminal window, navigate to the project directory and run:

   ```sh
   python backend/main.py
   ```

2. **Start the Streamlit Frontend:**

   In another terminal window, run:

   ```sh
   streamlit run streamlit_app.py
   ```

3. **Access the Application:**

   Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to interact with the UI.

### Using Docker

1. **Build & Run Containers:**

   ```sh
   docker-compose up --build
   ```

2. **Access the Services:**
   - Streamlit app: [http://localhost:8501](http://localhost:8501)
   - Backend service: as specified in your Docker configuration

## Configuration

The platform uses a YAML-based configuration system managed by the backend. The configuration file is loaded via `backend/utils/config.py` following the schema defined in the `Config` model. You can override default configurations through the Streamlit UI.

Example configuration parameters include:

- **Model Type:** e.g., FCMNIST, FNet, Adult, COVERTYPE, etc.
- **Dataset:** e.g., mnist, fashion_mnist, scikit-learn/adult-census-income, etc.
- **Federated Settings:** Number of clients, sampled clients, global/local epochs, learning rate, batch size, etc.
- **Security Settings:** Type of attack (SIGN_FLIP, MIN_MAX, LABEL_FLIP, etc.), number of attackers, aggregation strategy (FEDAVG, KRUM, MEDIAN, etc.)
- **Differential Privacy:** Enable or disable local DP-SGD through the configuration.

## Logging & Querying

- **Logging:**  
  The project uses Python’s logging module for real-time diagnostics and debugging. Logs are output to the console and can be redirected as needed.

- **Querying:**  
  Simulation results are stored in a SQL database. The platform provides an interactive query endpoint (accessible via the Streamlit app) for retrieving past results based on various filters (e.g., dataset, attack type, aggregation strategy).

## Contributing

Contributions are welcome! If you have ideas for new features or improvements:
- Open an Issue to discuss your ideas.
- Submit a Pull Request with your improvements.
- Follow the existing coding style and include clear commit messages.

## License

This project is licensed under [MIT License](LICENSE).

---

FL Simulator empowers researchers and practitioners to explore the intersection of federated learning and adversarial robustness. With its modular design and extensive configuration options, it serves as a powerful tool for simulating complex learning scenarios and evaluating defense strategies in a privacy-preserving manner.
