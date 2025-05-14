import streamlit as st
from logic.orchestrator import process_question
from data.log_reader import read_data_logs, read_error_logs, get_log_statistics_by_type
from data.logger import log_error
import pandas as pd

async def start_interface():
    #try:
        MAX_CHARS = 150
        st.set_page_config(page_title="RAG System", layout="wide")
        page = st.sidebar.selectbox("Navigation", ["Queries", "History", "Logs", "Statistics"])
        st.title("RAG System")
        
        if page == "Queries":
            st.warning("Questions will be stored for data analysis. Do not enter personal information(name, password, credit card numbers, email, address, etc.).")

            st.subheader("What kind of questions can you ask?")
            st.markdown("""
                **Examples:**
                - "What challenges do software developers face?"
                - "What advancements have been made in software product lines?"
                - "What is the impact of model variants comparison on EMF-based model variants?"
                - "What are the main challenges in software architecture?"
                - "How many stakeholders are affected by the lack of software evolution history?"
            """)

            if "history" not in st.session_state:
                st.session_state.history = []

            question = st.text_input("Enter your question:", value="")
            if len(question) > MAX_CHARS:
                st.warning(f"Your question is too long. Please keep it under {MAX_CHARS} characters.")
            if st.button("Ask", disabled=len(question) > MAX_CHARS):
                if question:
                    with st.spinner("Processing..."):
                            response_placeholder = st.empty()
                            #try:
                            response = await process_question(question)
                            st.session_state.history.append({
                                "question": question,
                                "response": response
                            })
                            response_placeholder.write(response) 
                            # except Exception as e:
                            #     response_placeholder.error("An unexpected error occurred. Try again.")
                                                    
                else:
                    st.warning("Please enter a question.")
                

        elif page == "History":
            st.subheader("Query History")
            if st.session_state.get("history"):
                st.markdown("---")

                for idx, entry in enumerate(reversed(st.session_state.history)):
                    with st.expander(f"Question #{len(st.session_state.history) - idx}: {entry['question']}"):
                        st.markdown(f"**Answer:**\n> {entry['response']}")

                if st.button("Clear history"):
                    st.session_state.history = []
                    st.success("History cleared successfully.")
            else:
                st.info("You haven't asked any questions yet.")

        elif page == "Logs":
            st.subheader("Logs")
            log_category = st.radio("Select log type", ["Queries", "LLM Calls", "Embeddings", "Database", "Errors"])

            logs_by_type = read_data_logs()
            error_logs = read_error_logs()

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

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No logs available for this category.")

        elif page == "Statistics":
            stats_by_type = get_log_statistics_by_type()

            st.markdown("Log Statistics by Type")
            for log_type, stat in stats_by_type.items():
                with st.expander(f"Log Type: {log_type}"):
                    if "task_name" in stat["df"].columns:
                        
                        task_names = stat["df"]["task_name"].unique()
                        selected_task = st.selectbox(f"Select Task for {log_type}", options=task_names)

                       
                        filtered_df = stat["df"][stat["df"]["task_name"] == selected_task]
                        
                        
                        task_stat = {
                            "total_cost": filtered_df["cost"].sum() if "cost" in filtered_df else None,
                            "avg_cost": filtered_df["cost"].mean() if "cost" in filtered_df else None,
                            "avg_duration_s": filtered_df["log_duration_sec"].mean() if "log_duration_sec" in filtered_df else None,
                            "count": len(filtered_df),
                            "df": filtered_df
                        }
                        
                        
                        col1, col2= st.columns(2)
                        col1.metric("Total Entries", task_stat["count"])
                        col2.metric("Avg Duration (s)", f"{task_stat['avg_duration_s']:.2f}")
                        
                        if task_stat["avg_cost"] is not None:
                            col1, col2 = st.columns(2)
                            col1.metric("Average Cost (€)", f"{task_stat['avg_cost']:.4f}")
                            col2.metric("Total Cost (€)", f"{task_stat['total_cost']:.4f}")
                        

                        
                        if "cost" in filtered_df.columns:
                            st.line_chart(filtered_df[["cost"]].reset_index(drop=True), use_container_width=True)

                    else:
                        col1, col2 = st.columns(2)
                        col1.metric("Total Entries", stat["count"])                       
                        col2.metric("Avg Duration (s)", f"{stat['avg_duration_s']:.2f}")

                        if stat["avg_cost"] is not None:
                            col1, col2= st.columns(2)
                            col1.metric("Average Cost (€)", f"{stat['avg_cost']:.4f}")
                            col2.metric("Total Cost (€)", f"{stat['total_cost']:.4f}")
                        
                        if "cost" in stat["df"].columns:
                            st.line_chart(stat["df"][["cost"]].reset_index(drop=True), use_container_width=True)

    # except Exception as e:
    #     log_error("GUIError", {
    #         "error": str(e), 
    #     })
