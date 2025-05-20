import streamlit as st
from logic.orchestrator import process_question
from logic.logs_service import get_logs, get_log_statistics_by_type
from logs.logger import log_error
import pandas as pd

async def start_interface() -> None:
    """
    Launches the Streamlit-based user interface for the RAG (Retrieval-Augmented Generation) system.

    This interface provides four main functionalities accessible through a sidebar:
        1. Queries: Submit natural language questions and get answers via the RAG backend.
        2. History: View a session-based history of previously asked questions and their responses.
        3. Logs: Explore backend system logs, categorized by type (e.g., queries, LLM calls, errors).
        4. Statistics: Visual analysis and metrics based on log data, including costs and duration.
    """
    try:
        MAX_CHARS = 150 #Define a character limit for the input question

        #Configure Streamlit UI settings
        st.set_page_config(page_title="RAG System", layout="wide")

        #Sidebar navigation to switch between pages
        page = st.sidebar.selectbox("Navigation", ["Queries", "History", "Logs", "Statistics"])

        #Main page title
        st.title("RAG System")
        
        #----------------- Page 1: Queries -----------------
        if page == "Queries":

            #Warning message about data privacy
            st.warning("Questions will be stored for data analysis. Do not enter personal information(name, password, credit card numbers, email, address, etc.).")

            #Example prompts to guide the user
            st.subheader("What kind of questions can you ask?")
            st.markdown("""
                **Examples:**
                - "What challenges do software developers face?"
                - "What advancements have been made in software product lines?"
                - "What problems are related?"
                - "What are the main challenges in software architecture?"
                - "How many stakeholders are affected by the lack of software evolution history?"
            """)

            #Initialize history in session if not already present
            if "history" not in st.session_state:
                st.session_state.history = []

            #Input field for the user question
            question = st.text_input("Enter your question:", value="")

            #Warn if question is too long
            if len(question) > MAX_CHARS:
                st.warning(f"Your question is too long. Please keep it under {MAX_CHARS} characters.")

            #Ask button triggers the async backend processing
            if st.button("Ask", disabled=len(question) > MAX_CHARS):
                if question:
                    with st.spinner("Processing...", show_time=True):
                            response_placeholder = st.empty() #Placeholder to dynamically display response
                            try:
                                #Call the backend to process the question
                                response = await process_question(question.lower().strip())

                                #Save question-response to session history
                                st.session_state.history.append({
                                    "question": question,
                                    "response": response
                                })
                                response_placeholder.write(response) 
                            except Exception as e:
                                #Display error if something goes wrong
                                response_placeholder.error("An unexpected error occurred. Try again.")
                                                    
                else:
                    st.warning("Please enter a question.")
                
        #----------------- Page 2: History -----------------
        elif page == "History":
            st.subheader("Query History")
            if st.session_state.get("history"):
                st.markdown("---")

                #Show questions and responses
                for idx, entry in enumerate(st.session_state.history):
                    with st.expander(f"Question #{len(st.session_state.history) - idx}: {entry['question']}"):
                        st.markdown(f"**Answer:**\n> {entry['response']}")

                #Option to clear history
                if st.button("Clear history"):
                    st.session_state.history = []
                    st.success("History cleared successfully.")
            else:
                st.info("You haven't asked any questions yet.")

        #----------------- Page 3: Logs -----------------
        elif page == "Logs":
            st.subheader("Logs")

            #Select log type to display
            log_category = st.radio("Select log type", ["Queries", "LLM Calls", "Embeddings", "Database", "Errors"])

            #Retrieve logs from system
            logs_by_type, error_logs = get_logs()

            #Choose data to display based on selected category
            if log_category == "Queries":
                data = logs_by_type["register_query"]
            elif log_category == "LLM Calls":
                data = logs_by_type["llm_call"]
            elif log_category == "Embeddings":
                data = logs_by_type["embedding"]
            elif log_category == "Database":
                data = logs_by_type["database"]
            elif log_category == "Errors":
                data = error_logs

            #Show logs in a table or show info message
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No logs available for this category.")

        #----------------- Page 4: Statistics -----------------
        elif page == "Statistics":
            stats_by_type = get_log_statistics_by_type()

            if stats_by_type:
                st.markdown("Log Statistics by Type")

                for log_type, stat in stats_by_type.items():
                    #Separate logs by type
                    with st.expander(f"Log Type: {log_type}"):
                        df = stat["df"]

                        if "tasks" in stat:
                            #Choose what log task to view
                            selected_task = st.selectbox(f"Select Task for {log_type}", options=list(stat["tasks"].keys()))
                            task_stat = stat["tasks"][selected_task]

                            col1, col2 = st.columns(2)
                            col1.metric("Total Entries", task_stat["count"])
                            col2.metric("Avg Duration (s)", f"{task_stat['avg_duration_s']:.2f}")

                            #If the task has costs, show them
                            if task_stat["avg_cost"] is not None:
                                col1, col2 = st.columns(2)
                                col1.metric("Average Cost ($)", f"{task_stat['avg_cost']:.4f}")
                                col2.metric("Total Cost ($)", f"{task_stat['total_cost']:.4f}")

                            if "cost" in task_stat["df"].columns:
                                st.line_chart(task_stat["df"][["cost"]].reset_index(drop=True), use_container_width=True)
                        else:
                            col1, col2 = st.columns(2)
                            col1.metric("Total Entries", stat["count"])
                            col2.metric("Avg Duration (s)", f"{stat['avg_duration_s']:.2f}")
                            
                            #If the log has costs, show them
                            if stat["avg_cost"] is not None:
                                col1, col2 = st.columns(2)
                                col1.metric("Average Cost ($)", f"{stat['avg_cost']:.4f}")
                                col2.metric("Total Cost ($)", f"{stat['total_cost']:.4f}")

                            if "cost" in df.columns:
                                st.line_chart(df[["cost"]].reset_index(drop=True), use_container_width=True)
            else:
                st.info("No statistics available.")

    except Exception as e:
        log_error("GUIError", {
            "error": str(e), 
        })
