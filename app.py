import streamlit as st
from agent import create_workflow
from dotenv import load_dotenv
import os
import time

# Load environment
load_dotenv()

# UI Setup
st.set_page_config(page_title="Research Agent", layout="wide")
st.title("🔍 Research Agent + ✍️ Draft Agent")

# Sidebar for API configuration
with st.sidebar:
    st.header("Configuration")
    tavily_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY"))
    mistral_key = st.text_input("Mistral API Key", type="password", value=os.getenv("MISTRAL_API_KEY"))
    if st.button("Update Keys"):
        os.environ["TAVILY_API_KEY"] = tavily_key
        os.environ["MISTRAL_API_KEY"] = mistral_key
        st.success("API keys updated!")

# Main form
with st.form("agent_form"):
    query = st.text_area("Research Topic", placeholder="e.g. Latest AI advancements in 2024")
    submitted = st.form_submit_button("Generate Report")

if submitted and query:
    with st.status("Running research pipeline...", expanded=True) as status:
        try:
            # Initialize workflow
            workflow = create_workflow()
            
            # Research step
            st.write("🔍 Conducting research...")
            research_state = {"user_input": query}
            result = workflow.invoke(research_state)
            time.sleep(1)  # Simulate processing
            
            # Draft step
            st.write("✍️ Writing report...")
            time.sleep(1)
            
            # Display results
            status.update(label="Pipeline complete!", state="complete")
            
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                st.success("Report generated successfully!")
                st.divider()
                st.subheader("Research Report")
                st.write(result["generated_draft"])
                
                # Download button
                st.download_button(
                    label="Download Report",
                    data=result["generated_draft"],
                    file_name=f"{query[:20]}_report.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")