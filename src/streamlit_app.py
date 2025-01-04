import os

import streamlit as st
import yaml

from main import process_data


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_file}")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML configuration: {e}")
        st.stop()


def main():
    st.title("Cryptocurrency Data Processing")
    st.write("This Streamlit app allows you to fetch and process cryptocurrency data.")

    config_file = st.text_input("Configuration File Path", "configs/config.yaml")
    if not os.path.exists(config_file):
        st.error("Configuration file not found. Please provide a valid path.")
        st.stop()

    if st.button("Run Data Processing"):
        try:
            process_data()
            st.success("Data processing completed successfully!")
        except Exception as e:
            st.error(f"Error during data processing: {e}")


if __name__ == "__main__":
    main()
