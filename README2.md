# 🚀 FL Simulator

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A comprehensive federated learning simulation platform for research and experimentation*

<img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" width="120" height="40" alt="PyTorch"/>
<img src="https://docs.ray.io/en/latest/_static/ray_logo.png" width="100" height="40" alt="Ray"/>
<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="40" height="40" alt="Streamlit"/>
<img src="https://flask.palletsprojects.com/en/2.3.x/_images/flask-horizontal.png" width="120" height="40" alt="Flask"/>

</div>

---

## 🌟 Overview

FL Simulator is a cutting-edge federated learning simulation platform designed for both academic research and industrial applications. Built with modern technologies and best practices, it enables researchers to explore the intersection of federated learning and adversarial robustness through scalable, distributed simulations.

## ✨ Key Features

### 🔥 **Scalable Architecture**
- **Distributed Training**: Leverages Ray for parallel client training to minimize run-times
- **Modular Design**: Easily extensible with custom attack methods, defense mechanisms, datasets, and models
- **High Performance**: Optimized for both CPU and GPU environments

### 🛡️ **Comprehensive Security Testing**
- **Advanced Attack Schemes**:
  - Label Flip Attacks
  - Min-Max & Min-Sum Attacks
  - Krum & Trim Attacks
  - Gaussian Noise Attacks
  - Sign Flip Attacks
  - No Attack (clean training baseline)

- **Robust Defense Mechanisms**:
  - **FedAvg** - Standard federated averaging
  - **KRUM** - Byzantine-robust aggregation
  - **MEDIAN** - Coordinate-wise median
  - **TRIM_MEAN** - Trimmed mean aggregation
  - **KeTS** - Trust-score based defense
  - **FLTRUST** - Server-assisted robust aggregation
  - Custom variants for research purposes

### 🔐 **Privacy Protection**
- **Differential Privacy**: Built-in integration with [Opacus](https://opacus.ai/) for DP-SGD
- **Configurable Privacy Budgets**: Fine-tune privacy-utility trade-offs
- **Local Privacy**: Client-side differential privacy implementation

### 📊 **Rich Dataset Support**
- **Tabular Data**: `kdd-cup-1999`, `adult-census-income`, `covtype`
- **Image Data**: `MNIST`, `Fashion-MNIST`, `CIFAR-10`
- **Flexible Partitioning**: IID and Non-IID data distribution strategies

### 🎯 **Interactive Interface**
- **Streamlit Frontend**: User-friendly web interface for configuration and monitoring
- **Real-time Visualization**: Live training metrics and performance dashboards
- **Historical Analysis**: Query and analyze past experiment results

## 🏗️ Architecture

<div align="center">

```mermaid
graph TB
    A[Streamlit Frontend] --> B[Flask Backend]
    B --> C[Ray Cluster]
    C --> D[Client 1]
    C --> E[Client 2]
    C --> F[Client N]
    B --> G[SQL Database]
    B --> H[Configuration Manager]
    I[Aggregation Layer] --> J[Defense Mechanisms]
    I --> K[Attack Simulation]
    
    style A fill:#FF4B4B,stroke:#fff,color:#fff
    style B fill:#000,stroke:#fff,color:#fff
    style C fill:#028CF0,stroke:#fff,color:#fff
    style G fill:#D71F00,stroke:#fff,color:#fff
```

</div>

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
   git clone <repository_url>
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

### 🏃‍♂️ Running Locally

1. **Start the Backend**
   ```bash
   python backend/main_bp.py
   ```

2. **Launch the Frontend**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Application**
   - Open your browser and navigate to `http://localhost:8501`

### 🐳 Docker Deployment

```bash
docker-compose up --build
```

**Access Points:**
- Streamlit App: `http://localhost:8501`
- Backend API: As configured in docker-compose.yml

## ⚙️ Configuration

The platform uses a YAML-based configuration system with comprehensive parameter control:

### 🎛️ **Model Configuration**
- **Architectures**: FCMNIST, FNet, Adult, COVERTYPE, TinyVGG
- **Optimization**: SGD, Adam, AdamW with configurable hyperparameters
- **Regularization**: Weight decay, momentum, learning rate scheduling

### 📈 **Training Parameters**
- **Federated Settings**: Number of clients, sampling strategies, epochs
- **Learning Configuration**: Learning rates, batch sizes, local updates
- **Privacy Settings**: DP-SGD parameters, noise multipliers

### 🔧 **Security Configuration**
- **Attack Types**: Configurable attack intensity and target selection
- **Defense Strategies**: Aggregation method selection and tuning
- **Trust Management**: Dynamic trust score updates and thresholds

## 📊 Technology Stack

<div align="center">

| Component | Technology | Purpose |
|-----------|------------|---------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | **PyTorch** | Deep learning framework |
| ![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white) | **Ray** | Distributed computing |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | **Streamlit** | Interactive frontend |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) | **Flask** | REST API backend |
| ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white) | **SQLAlchemy** | Database ORM |
| ![Opacus](https://img.shields.io/badge/Opacus-4285F4?style=flat&logoColor=white) | **Opacus** | Differential privacy |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | **NumPy** | Numerical computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | **Pandas** | Data manipulation |

</div>

## 📈 Performance Metrics

The platform provides comprehensive logging and analysis capabilities:

- **Training Metrics**: Loss curves, accuracy progression, convergence analysis
- **Security Metrics**: Attack success rates, defense effectiveness
- **System Metrics**: Training time, resource utilization, scalability benchmarks
- **Privacy Metrics**: Privacy budget consumption, utility preservation

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

### Development Guidelines
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation as needed

## 📚 Documentation

- **API Reference**: Detailed endpoint documentation
- **Configuration Guide**: Complete parameter reference
- **Research Papers**: Implementation details and experimental results
- **Tutorials**: Step-by-step usage examples

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Ray Project** for enabling distributed computing
- **Streamlit** for the intuitive web interface
- **Opacus Team** for differential privacy tools
- **Research Community** for federated learning advances

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Project Wiki
- **Email**: [Contact Email](mailto:contact@example.com)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

*FL Simulator - Empowering federated learning research through advanced simulation capabilities*

![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-❤️-brightgreen.svg)

</div>