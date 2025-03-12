# FL Simulator

# FL Simulator

FL Simulator is a federated learning simulation platform. The project consists of a Python-based backend and a [Streamlit](https://streamlit.io) frontend, enabling you to configure parameters, run simulations, and execute queries interactively. It also offers various attack and defense schemes to help study the robustness and security of federated learning settings.

## Project Structure

- **backend/** – Python backend code including simulation logic and configuration handling ([`backend/utils/config.py`](c:\Users\39340\FL_Simulator\backend\utils\config.py)).
- **streamlit_app.py** – The Streamlit application for interacting with the simulation ([`streamlit_app.py`](c:\Users\39340\FL_Simulator\streamlit_app.py)).
- **docker-compose.yml** – Docker Compose configuration to orchestrate the services.
- **Dockerfile-streamlit** – Dockerfile to build the Streamlit container.
- **requirements.txt** – List of required Python packages.

## Getting Started

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/) (if running with Docker)
- pip

### Local Setup

1. **Clone the repository:**

   ```sh
   git clone <repository_url>
   cd FL_Simulator
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

### Running the Project Locally

1. **Start the backend:**

   Launch your Python backend (for example, if your backend entry point is `main.py`):

   ```sh
   python backend/main.py
   ```

2. **Start the Streamlit app:**

   In a separate terminal, run:

   ```sh
   streamlit run streamlit_app.py
   ```

Access the Streamlit interface in your browser at `http://localhost:8501`.

## Docker Setup

You can run the entire project using Docker.

1. **Build and run the containers using Docker Compose:**

   ```sh
   docker-compose up --build
   ```

   This command will:
   - Launch the backend service.
   - Build and launch the Streamlit app using [Dockerfile-streamlit](Dockerfile-streamlit).

2. **Access the services:**

   - The Streamlit app will be available at `http://localhost:8501`.
   - The backend service should be running as specified in its Docker configuration.

## Configuration

The backend configuration is managed via a YAML file and loaded in [`backend/utils/config.py`](c:\Users\39340\FL_Simulator\backend\utils\config.py) using the `load_config` function. Ensure that any overrides follow the schema defined in the `Config` model.

## Usage

- **Set Parameters:** Configure simulation parameters using the form in the Streamlit app.
- **Run Simulation:** Start simulation runs from the "Run Simulation" section in the app.
- **Execute Query:** Use the interactive query page to inspect simulation results.

## Logging

The project uses Python's logging module. Logs will be output to the console for easy debugging.


## Contributions

Feel free to open issues or pull requests if you'd like to contribute!
