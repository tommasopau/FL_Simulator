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
        st.subheader("Federated Learning Parameters")
        aggregation_strategy = st.selectbox(
            "Aggregation Strategy",
            ["FLTRUST", "FEDAVG", "KRUM", "MEDIAN", "TRIM_MEAN", "KeTS"]
        )
        alpha = st.number_input("Alpha", value=1.0, step=0.1)
        attack = st.selectbox(
            "Attack",
            ["LABEL_FLIP", "NO_ATTACK", "MIN_MAX", "MIN_SUM", "KRUM", "TRIM", "GAUSSIAN", "SIGN_FLIP"]
        )
        batch_size = st.number_input("Batch Size", value=64, step=1)
        dataset = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
        global_epochs = st.number_input("Global Epochs", value=20, step=1)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        local_dp = st.checkbox("Local DP-SGD", value=False)
        local_epochs = st.number_input("Local Epochs", value=4, step=1)
        num_attackers = st.number_input("Number of Attackers", value=10, step=1)
        num_clients = st.number_input("Number of Clients", value=50, step=1)
        partition_type = st.selectbox("Partition Type", ["iid", "non-iid"])
        sampled_clients = st.number_input("Sampled Clients", value=40, step=1)
        seed = st.number_input("Seed", value=42, step=1)
        
        st.subheader("Model Parameters")
        model_type = st.selectbox("Model Type", ["FCMNIST", "ZalandoCNN"])
        
        submitted = st.form_submit_button("Start Simulation")
    
    if submitted:
        # Build configuration dictionary based on selected parameters
        config = {
            "federated_learning": {
                "aggregation_strategy": aggregation_strategy,
                "alpha": alpha,
                "attack": attack,
                "batch_size": batch_size,
                "dataset": dataset,
                "global_epochs": global_epochs,
                "learning_rate": learning_rate,
                "local_DP_SGD": local_dp,
                "local_epochs": local_epochs,
                "num_attackers": num_attackers,
                "num_clients": num_clients,
                "partition_type": partition_type,
                "sampled_clients": sampled_clients,
                "seed": seed,
            },
            "model": {
                "type": model_type
            }
        }
        st.info("Posting selected configuration to simulation endpoint...Waiting for simulation..")
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
                if "trustscores" in data:
                    trust_scores = data["trustscores"]
                    trust_df = pd.DataFrame(list(trust_scores.items()), columns=["Client ID", "Trust Score"])
                    st.bar_chart(trust_df.set_index("Client ID"))
            else:
                st.error("Failed to start simulation.")
        except Exception as e:
            st.error(f"Error: {e}")
            
elif page == "Execute Query":
    st.header("Execute SQL Query")
    # Initialize session state for filters if not already present
    if 'filters' not in st.session_state:
        st.session_state['filters'] = []
    
    filters = st.session_state.filters  # Reference the session state filters
    
    # Define available options for filtering
    AVAILABLE_OPTIONS = {
        "dataset": ["mnist", "fashion_mnist"],
        "attack": ["MIN_MAX", "MIN_SUM", "KRUM", "TRIM", "NO_ATTACK", "LABEL_FLIP", "GAUSSIAN", "SIGN_FLIP"],
        "aggregation_strategy": ["FEDAVG", "KRUM", "MEDIAN", "TRIM_MEAN", "KeTS", "FLTRUST"],
        "partition_type": ["iid", "non-iid"],
        "model_type": ["FCMNIST", "ZalandoCNN"],
    }
    
    st.subheader("Filter Settings")
    
    # Dropdown for selecting a column
    field = st.selectbox("Select Column", options=list(AVAILABLE_OPTIONS.keys()))
    
    if field:
        # Dropdown for selecting an operator
        operator = st.selectbox(f"Select Operator for '{field}'", options=["eq"])
        # Input field for the value
        value = st.selectbox(f"Select value for '{field}'", options=AVAILABLE_OPTIONS.get(field, []))
        
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
                        st.write("Query Results:", result)
                    else:
                        st.write("No results found.")
                else:
                    st.error(f"Error: {response.json().get('error')}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please add at least one filter before submitting the query.")