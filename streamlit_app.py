import streamlit as st
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sidebar Navigation: Two pages now.
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Run Simulation", "Execute Query"])

if page == "Run Simulation":
    st.header("Run Federated Learning Simulation")
    
    with st.form("simulation_params_form"):
        st.subheader("Model Parameters")
        model_type = st.selectbox("Model Type", ["MNISTCNN", "FCMNIST", "ZalandoCNN", "ResNetCIFAR10", "TinyVGG", "KDDSimpleNN"])
        
        st.subheader("Dataset Parameters")
        dataset = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
        num_clients = st.number_input("Number of Clients", value=50, step=1, min_value=2)
        alpha = st.number_input("Alpha", value=1.0, step=0.1, min_value=0.05, max_value=10.0)
        batch_size = st.number_input("Batch Size", value=64, step=1, min_value=1)
        partition_type = st.selectbox("Partition Type", ["iid", "non_iid"])
        
        st.subheader("Training Parameters")
        global_epochs = st.number_input("Global Epochs", value=20, step=1, min_value=1)
        local_epochs = st.number_input("Local Epochs", value=4, step=1, min_value=1)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f", min_value=0.0001)
        sampled_clients = st.number_input("Sampled Clients", value=40, step=1, min_value=1)
        
        # Advanced Training Parameters
        local_dp = st.checkbox("Local DP-SGD", value=False)
        fedprox = st.checkbox("FedProx", value=False)
        fedprox_mu = st.number_input("FedProx Mu", value=0.01, step=0.001, min_value=0.0)
        
        # Optimizer Parameters
        optimizer = st.selectbox("Optimizer", ["SGD", "Adam"])
        momentum = st.number_input("Momentum", value=0.9, step=0.1, min_value=0.0, max_value=1.0) 

        
        st.subheader("Attack Parameters")
        attack = st.selectbox(
            "Attack Type",
            ["no_attack", "label_flip", "min_max", "min_sum", "krum", "trim", "gaussian", "sign_flip"]
        )
        num_attackers = st.number_input("Number of Attackers", value=0, step=1, min_value=0) 
        
        st.subheader("System Parameters")
        seed = st.number_input("Seed", value=42, step=1)
        
        st.subheader("Aggregation Parameters")
        aggregation_strategy = st.selectbox(
            "Aggregation Strategy",
            ["fedavg", "fltrust", "krum", "median", "trim_mean", "KeTS"]
        )
        
        submitted = st.form_submit_button("Start Simulation")
    
    if submitted:
        # Validate sampled_clients doesn't exceed num_clients
        if sampled_clients > num_clients:
            st.error(f"Sampled clients ({sampled_clients}) cannot exceed total clients ({num_clients})")
        elif attack != "no_attack" and num_attackers > num_clients:
            st.error(f"Number of attackers ({num_attackers}) cannot exceed total clients ({num_clients})")
        elif attack != "no_attack" and num_attackers == 0:
            st.error("Number of attackers must be greater than 0 when an attack is specified")
        else:
            # Build configuration dictionary with new structure
            config = {
                "model": {
                    "type": model_type
                },
                "dataset": {
                    "dataset": dataset,
                    "num_clients": num_clients,
                    "alpha": alpha,
                    "batch_size": batch_size,
                    "partition_type": partition_type
                },
                "training": {
                    "global_epochs": global_epochs,
                    "local_epochs": local_epochs,
                    "learning_rate": learning_rate,
                    "sampled_clients": sampled_clients,
                    "local_DP_SGD": local_dp,
                    "fedprox": fedprox,
                    "fedprox_mu": fedprox_mu,
                    "optimizer": optimizer,
                    "momentum": momentum
                },
                "attack": {
                    "attack": attack,
                    "num_attackers": num_attackers
                },
                "system": {
                    "seed": seed
                },
                "aggregation": {
                    "aggregation_strategy": aggregation_strategy
                }
            }
            
            st.info("Posting selected configuration to simulation endpoint... Waiting for simulation...")
            
            # Display the configuration being sent
            with st.expander("View Configuration"):
                st.json(config)
            
            try:
                sim_response = requests.post("http://localhost:8000/simulation", json=config)
                if sim_response.status_code == 200:
                    st.balloons()
                    data = sim_response.json()
                    status = data.get("status", "No status provided.")
                    accuracy = data.get("accuracy", "No accuracy provided.")
                    st.subheader("Simulation Results")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Accuracy:** {accuracy}")
                    
                    # Display configuration used
                    if "config" in data:
                        with st.expander("Configuration Used"):
                            st.json(data["config"])
                    
                    # Display trust scores if available
                    if "trustscores" in data:
                        trust_scores = data["trustscores"]
                        trust_df = pd.DataFrame(list(trust_scores.items()), columns=["Client ID", "Trust Score"])
                        st.subheader("Trust Scores")
                        st.bar_chart(trust_df.set_index("Client ID"))
                        
                else:
                    st.error(f"Failed to start simulation. Status: {sim_response.status_code}")
                    if sim_response.text:
                        st.error(f"Error details: {sim_response.text}")
                        
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Make sure the backend server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Simulation request failed")
            
elif page == "Execute Query":
    st.header("Execute SQL Query")
    # Initialize session state for filters if not already present
    if 'filters' not in st.session_state:
        st.session_state['filters'] = []
    
    filters = st.session_state.filters  # Reference the session state filters
    
    # Updated available options to match new config structure
    AVAILABLE_OPTIONS = {
        "dataset": ["mnist", "fashion_mnist"],
        "attack": ["NO_ATTACK", "LABEL_FLIP", "MIN_MAX", "MIN_SUM", "KRUM", "TRIM", "GAUSSIAN", "SIGN_FLIP"],
        "aggregation_strategy": ["FEDAVG", "FLTRUST", "KRUM", "MEDIAN", "TRIM_MEAN", "KeTS"],
        "partition_type": ["iid", "non_iid"],
        "model_type": ["MNISTCNN", "FCMNIST", "ZalandoCNN", "ResNetCIFAR10", "TinyVGG", "KDDSimpleNN"],
        "optimizer": ["SGD", "Adam"],
    }
    
    st.subheader("Filter Settings")
    
    # Dropdown for selecting a column
    field = st.selectbox("Select Column", options=list(AVAILABLE_OPTIONS.keys()))
    
    if field:
        # Dropdown for selecting an operator
        operator = st.selectbox(f"Select Operator for '{field}'", options=["eq", "ne", "gt", "lt", "gte", "lte"])
        
        # Input field for the value
        if field in AVAILABLE_OPTIONS:
            value = st.selectbox(f"Select value for '{field}'", options=AVAILABLE_OPTIONS[field])
        else:
            value = st.text_input(f"Enter value for '{field}'")
        
        # Add filter button
        if st.button("Add Filter"):
            if value:
                filters.append({"field": field, "operator": operator, "value": value})
                st.session_state.filters = filters
                st.success(f"Filter added: {field} {operator} {value}")
            else:
                st.warning("Please enter a value before adding the filter.")
    
    # Show the current filters with remove option
    if filters:
        st.write("Current filters:")
        for idx, f in enumerate(filters):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{f['field']} {f['operator']} {f['value']}")
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    filters.pop(idx)
                    st.session_state.filters = filters
                    st.rerun()
    
    # Button to submit the query
    if st.button("Submit Query"):
        if filters:
            payload = {"filters": filters}
            try:
                response = requests.post('http://localhost:8000/query', json=payload)
                if response.status_code == 200:
                    result = response.json().get('result', [])
                    if result:
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(result)
                        st.subheader("Query Results")
                        st.dataframe(df)
                        
                        # Show summary statistics
                        if 'accuracy' in df.columns:
                            st.subheader("Summary Statistics")
                            st.write(f"**Number of results:** {len(df)}")
                            st.write(f"**Average accuracy:** {df['accuracy'].mean():.4f}")
                            st.write(f"**Best accuracy:** {df['accuracy'].max():.4f}")
                            st.write(f"**Worst accuracy:** {df['accuracy'].min():.4f}")
                    else:
                        st.write("No results found.")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Make sure the backend server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Query request failed")
        else:
            st.warning("Please add at least one filter before submitting the query.")