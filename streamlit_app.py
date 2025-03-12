import streamlit as st
import requests
import pandas as pd
import socketio
from backend.utils.constants import ALLOWED_FILTERS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Set Parameters", "Run Simulation", "Execute Query"])

if page == "Set Parameters":
    st.header("Set Simulation Parameters")
    with st.form("config_form"):
        aggregation_strategy = st.selectbox("Aggregation Strategy", ["FEDAVG", "KRUM", "MEDIAN", "TRIM_MEAN", "KeTS" , "FLTRUST"])
        attack = st.selectbox("Attack Type", ["MIN_MAX", "MIN_SUM", "KRUM", "TRIM", "NO_ATTACK","LABEL_FLIP","GAUSSIAN","SIGN_FLIP"])
        dataset = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
        model_type = st.selectbox("Model Type", ["FCMNIST","FNet"])
        alpha = st.number_input("Alpha", min_value=0.05, max_value=10.0, value=0.5)
        batch_size = st.number_input("Batch Size", min_value=1, value=64)
        global_epochs = st.number_input("Global Epochs", min_value=1, value=5)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, step=0.0001)
        local_epochs = st.number_input("Local Epochs", min_value=1, value=4)
        num_attackers = st.number_input("Number of Attackers", min_value=0, value=10)
        num_clients = st.number_input("Number of Clients", min_value=2, value=50)
        partition_type = st.selectbox("Partition Type", ["iid", "non-iid"])
        sampled_clients = st.number_input("Sampled Clients", min_value=1, value=40)
        seed = st.number_input("Seed", value=42)
        local_DP_SGD = st.checkbox("Enable Local DP-SGD", value=False)
        
        submit = st.form_submit_button("Submit Parameters")

    if submit:
        config = {
            "model": {
                "type": model_type
            },
            "federated_learning": {
                "aggregation_strategy": aggregation_strategy,
                "alpha": alpha,
                "attack": attack,
                "batch_size": batch_size,
                "dataset": dataset,
                "global_epochs": global_epochs,
                "learning_rate": learning_rate,
                "local_epochs": local_epochs,
                "num_attackers": num_attackers,
                "num_clients": num_clients,
                "partition_type": partition_type,
                "sampled_clients": sampled_clients,
                "seed": seed,
                "local_DP_SGD": local_DP_SGD
            }
        }

        response = requests.post("http://localhost:8000/api/config", json=config)
        if response.status_code == 200:
            st.success("Configuration submitted successfully.")
        else:
            st.error("Failed to submit configuration.")

elif page == "Run Simulation":
    st.header("Run Federated Learning Simulation")
    
    if st.button("Start Simulation"):
        st.success("Training started", icon="🚀")
        try:
            sim_response = requests.get("http://localhost:8000/start")
            if sim_response.status_code == 200:
                st.balloons()

                data = sim_response.json()
                status = data.get("status", "No status provided.")
                accuracy = data.get("accuracy", "No accuracy provided.")
                
                if "trustscores" in data:
                    trust_scores = data["trustscores"]
                    trust_df = pd.DataFrame(list(trust_scores.items()), columns=["Client ID", "Trust Score"])
                    
                    # Display simulation results
                    st.subheader("Simulation Results")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Accuracy:** {accuracy}")
                    
                    # Plot trust scores as a bar chart
                    st.bar_chart(trust_df.set_index("Client ID"))
                else:
                    st.success(f"Status: {status}")
                    st.info(f"Accuracy: {accuracy}")
            else:
                st.error("Failed to start simulation.")
        except Exception as e:
            st.error(f"Connection Error: {e}")
    
elif page == "Execute Query":
    st.header("Execute SQL Query")

    # Initialize session state for filters if not already present
    if 'filters' not in st.session_state:
        st.session_state['filters'] = []

    filters = st.session_state.filters  # Reference the session state filters

    # Add the AVAILABLE_OPTIONS dictionary
    AVAILABLE_OPTIONS = {
        "dataset": ["mnist", "fashion_mnist"],
        "attack": ["MIN_MAX", "MIN_SUM", "KRUM", "TRIM", "NO_ATTACK","LABEL_FLIP","GAUSSIAN","SIGN_FLIP"],
        "aggregation_strategy": ["FEDAVG", "KRUM", "MEDIAN", "TRIM_MEAN", "KeTS","FLTRUST"],
        "partition_type": ["iid", "non-iid"],
        "model_type": ["FCMNIST", "ZalandoCNN"],
        # Add more fields and their possible values as needed
    }

    # Add filters for the query
    st.subheader("Filter Settings")
    
    # Dropdown for selecting a column
    field = st.selectbox("Select Column", options=list(ALLOWED_FILTERS.keys()))
    
    if field:
        # Dropdown for selecting an operator based on the selected field
        operator = st.selectbox(f"Select Operator for '{field}'", options=ALLOWED_FILTERS[field])
        
        # Input field for the value based on the selected operator
        if ALLOWED_FILTERS[field] == ["eq"]:
            value = st.selectbox(f"Select value for '{field}'", options=AVAILABLE_OPTIONS.get(field, []))
        elif "eq" in ALLOWED_FILTERS[field]:
            value = st.text_input(f"Enter value for '{field}'")
        else:
            value = st.number_input(f"Enter value for '{field}'", value=0.0)
        
        # Add filter button
        if st.button("Add Filter"):
            if value:  # Check that the user has entered a value
                # Add the filter to the session state
                filters.append({"field": field, "operator": operator, "value": value})
                st.session_state.filters = filters  # Store the updated filters in session state
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
            # Create a request payload with the filters
            payload = {"filters": filters}
            
            try:
                # Send the query to the API
                response = requests.post('http://localhost:8000/api/query', json=payload)
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