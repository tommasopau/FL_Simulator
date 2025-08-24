# 🚀 FL Simulator

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

**A cutting-edge federated learning simulation platform for academic research**

</div>

---

## 🌟 Overview

FL Simulator is a cutting-edge federated learning simulation platform designed for academic research. Built with modern technologies and best practices, it enables researchers to explore the intersection of federated learning and adversarial robustness through scalable, distributed simulations.

## ✨ Key Features

### 🔥 **Scalable Architecture**

- **Distributed Training**: Leverages Ray for parallel client training to minimize run-times
- **Modular Design**: Easily extensible with custom attack methods, defense mechanisms, datasets, and models
- **High Performance**: Optimized for both CPU and GPU environments

### 🛡️ **Comprehensive Security Testing**

- **Advanced Attack Schemes**:

  - Label Flip Attacks (https://arxiv.org/abs/1206.6389)
  - Min-Max & Min-Sum Attacks (https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/)
  - Krum & Trim Attacks (https://www.usenix.org/conference/usenixsecurity20/presentation/fang)
  - Gaussian Noise Attacks
  - Sign Flip Attacks (https://ieeexplore.ieee.org/document/10725463)
  - No Attack (clean training baseline)

- **Robust Defense Mechanisms**:
  - **FedAvg** - Standard federated averaging (https://proceedings.mlr.press/v54/mcmahan17a.html)
  - **KRUM** - Byzantine-robust aggregation (https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
  - **MEDIAN** - Coordinate-wise median (https://proceedings.mlr.press/v80/yin18a.html)
  - **TRIM_MEAN** - Trimmed mean aggregation (https://proceedings.mlr.press/v80/yin18a.html)
  - **KeTS** - Trust-score based defense (https://arxiv.org/abs/2501.06729)
  - **FLTRUST** - Server-assisted robust aggregation (https://arxiv.org/abs/2012.13995)
  - Custom variants for research purposes

### 🔐 **Privacy Protection**

- **Differential Privacy**: Built-in integration with [Opacus](https://opacus.ai/) for DP-SGD

### 📊 **Rich Dataset Support**

- **Tabular Data**: `kdd-cup-1999`,
- **Image Data**: `MNIST`, `Fashion-MNIST`, `CIFAR-10`
- **Flexible Partitioning**: IID and Non-IID data distribution strategies

### 🎯 **Interactive Interface**

- **Streamlit Frontend**: User-friendly web interface for configuration and monitoring
- **Historical Analysis**: Query and analyze past experiment results

## 🚀 Quick Start

### Prerequisites

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

</div>

- Python 3.10+
- Docker (optional)
- 8GB+ RAM recommended

### 📦 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/FL_Simulator.git
   cd FL_Simulator
   ```

2. **Create Virtual Environment**

   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**

   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env file if needed (optional for local SQLite)
   ```

### 🏃‍♂️ Running Locally

1. **Start the Backend Server**

   ```bash
   cd backend
   python main_bp.py
   ```

2. **Launch the Streamlit Frontend** (in a new terminal)

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Application**
   - Open your browser and navigate to `http://localhost:8501`
   - The backend API runs on `http://localhost:8000`

### 🐳 Docker Deployment

```bash
docker-compose up --build
```

**Access Points:**

- Streamlit App: `http://localhost:8501`
- Backend API: As configured in docker-compose.yml

## ⚙️ Configuration

1. **YAML File**: Edit `config.yaml` directly
2. **Streamlit Interface**: Use the web UI for interactive configuration

## 🚧 Work in Progress

FL Simulator is actively under development.

### 🗄️ Database Configuration

Configure your database by editing the `.env` file:

```bash
# Database Type
DB_TYPE=local                           # Options: 'local' or 'azure'

# Azure SQL Configuration (when DB_TYPE=azure)
SQL_DRIVER=ODBC Driver 18 for SQL Server
SQL_SERVER={your-sql-server-name-here}
SQL_DATABASE={your-database-name-here}
SQL_USER={your-username-here}
SQL_PASSWORD={your-password-here}
```

**Local SQLite**: Set `DB_TYPE=local` (default)
**Azure SQL**: Set `DB_TYPE=azuresql` and configure SQL parameters

> **Note**: Azure SQL requires [Microsoft ODBC Driver 18 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) to be installed on your system.

## 📊 Technology Stack

<div align="center">

| Component                                                                                                | Technology     | Purpose                 |
| -------------------------------------------------------------------------------------------------------- | -------------- | ----------------------- |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)          | **PyTorch**    | Deep learning framework |
| ![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white)                      | **Ray**        | Distributed computing   |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)    | **Streamlit**  | Interactive frontend    |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)                | **Flask**      | REST API backend        |
| ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white) | **SQLAlchemy** | Database ORM            |
| ![Opacus](https://img.shields.io/badge/Opacus-4285F4?style=flat&logoColor=white)                         | **Opacus**     | Differential privacy    |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)                | **NumPy**      | Numerical computing     |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)             | **Pandas**     | Data manipulation       |

</div>

## 🧪 Research Applications

FL Simulator is designed for cutting-edge research in:

- **Federated Learning Algorithms**: New aggregation strategies and optimization methods
- **Adversarial Robustness**: Novel attack vectors and defense mechanisms
- **Privacy-Preserving ML**: Differential privacy and secure aggregation techniques
- **System Optimization**: Scalability and efficiency improvements

## 🤝 Contributing

We welcome contributions from the research community!

### How to Contribute

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**⭐ Star this repository if you find it useful!**

_FL Simulator - Empowering federated learning research through advanced simulation capabilities_

![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-❤️-brightgreen.svg)

</div>

## 🧪 Running Simulations for Research

For researchers who need a faster and more flexible way to run simulations, you can directly execute the `simulationResearch.py` file. This approach allows you to quickly modify the simulation logic and parameters without relying on the full frontend/backend setup.

### Steps to Run Simulations

1. **Edit the Configuration File**  
   Update the `config.yaml` file to set the desired parameters for your simulation, such as:

   - Model type
   - Dataset and partitioning strategy
   - Training parameters (e.g., global/local epochs, learning rate)
   - Attack type and number of attackers
   - Aggregation strategy

   Example:

   ```yaml
   model:
     type: ZalandoCNN

   dataset:
     dataset: fashion_mnist
     num_clients: 50
     alpha: 0.5
     batch_size: 64
     partition_type: non_iid

   training:
     global_epochs: 20
     local_epochs: 3
     learning_rate: 0.001
     sampled_clients: 40
     local_DP_SGD: false
     fedprox: false
     fedprox_mu: 0.01
     optimizer: SGD
     momentum: 0.9

   attack:
     attack: min_max
     num_attackers: 12

   system:
     seed: 42

   aggregation:
     aggregation_strategy: KeTS
   ```

2. **Run the Simulation Script**  
   Execute the `simulationResearch.py` file directly from the terminal:

   ```bash
   python simulationResearch.py
   ```

3. **View Results**

   - The simulation results, including the final accuracy and configuration details, will be saved to `results.txt` in the project directory.
   - If epoch-wise results are available, they will be saved to `simulation_epoch_results.txt`.

4. **Modify the Script for Custom Logic**  
   If needed, you can directly modify the `simulationResearch.py` file to implement custom logic, such as:

   - Adding new attack or defense mechanisms
   - Logging additional metrics
   - Changing the dataset or model dynamically

   Example: To log client-specific metrics, you can extend the `get_client_class_counts` function or add custom logging within the `main()` function.

### Why Use This Approach?

- **Faster Iterations**: No need to start the backend or frontend services.
- **Research Flexibility**: Easily modify the script to test new ideas or configurations.
- **Direct Control**: Full access to the simulation logic and parameters.

This method is ideal for researchers who want to focus on experimentation and development without the overhead of the full application stack.
