import streamlit as st
from logic.orchestrator import process_question
from data.log_reader import read_query_logs, read_error_logs
from data.logger import log_error
import pandas as pd

async def start_interface():
    st.set_page_config(page_title="RAG System", layout="wide")
    page = st.sidebar.selectbox("Navigation", ["Queries", "History", "Logs", "Statistics"])
    st.title("RAG System")
    
    if page == "Queries":
        st.warning("Questions will be stored for data analysis. Do not enter personal information.")
        if "history" not in st.session_state:
            st.session_state.history = []

        question = st.text_input("Enter your question:", value="")

        response_placeholder = st.empty()

        if st.button("Ask"):
            try:
                if question:
                    with st.spinner("Processing..."):
                        
                            response = await process_question(question)
                            st.session_state.history.append({
                                "question": question,
                                "response": response
                            })

                            response_placeholder.write(response)                         
                else:
                    st.warning("Please enter a question.")
            except Exception as e:
                response_placeholder.error("An unexpected error occurred.")
                log_error("GUIError", {
                    "error": str(e),})

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
        log_type = st.radio("Select log type", ["Queries", "Errors"])

        if log_type == "Queries":
            logs = read_query_logs()
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df)
            else:
                st.info("No query logs available.")
        else:
            logs = read_error_logs()
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df)
            else:
                st.info("No error logs available.")

    elif page == "Statistics":
        st.subheader("System Statistics")

        query_logs = read_query_logs()
        error_logs = read_error_logs()

        if query_logs:
            df = pd.DataFrame(query_logs)

            # Total query cost
            total_cost_queries = df["total_cost"].sum()

            # Handle errors with costs
            if error_logs:
                error_df = pd.DataFrame(error_logs)
                for col in ["cost_gpt", "cost_embed", "total_cost"]:
                    error_df[col] = pd.to_numeric(error_df.get(col, 0), errors='coerce')
                total_cost_errors = error_df["total_cost"].sum()
            else:
                total_cost_errors = 0

            total_cost_global = total_cost_queries + total_cost_errors

            # Convert numeric columns
            for col in ["time_elapsed", "cost_gpt", "cost_embed", "total_cost"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            st.markdown("### General Metrics")

            st.markdown("### Total Accumulated Cost")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Query Cost", f"{total_cost_queries:.4f}€")
            col2.metric("Total Error Cost", f"{total_cost_errors:.4f}€")
            col3.metric("Total Global Cost", f"{total_cost_global:.4f}€")

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Response Time", f"{df['time_elapsed'].mean():.2f} s")
            col2.metric("Average Total Cost", f"{df['total_cost'].mean():.4f}€")
            col3.metric("Logged Queries", len(df))

            st.markdown("### Component-wise Costs")
            st.bar_chart(df[["cost_gpt", "cost_embed"]].mean())

            st.markdown("### Total Cost Distribution")
            st.line_chart(df["total_cost"])

            st.markdown("### Top 5 Most Expensive Questions")
            top_cost = df.sort_values("total_cost", ascending=False).head(5)[["question", "total_cost"]]
            st.table(top_cost)

        else:
            st.info("Not enough query logs to display statistics.")

        st.markdown("---")
        st.markdown("### Most Common Errors")

        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_counts = error_df["error_type"].value_counts().reset_index()
            error_counts.columns = ["Error Type", "Frequency"]
            st.bar_chart(error_counts.set_index("Error Type"))
        else:
            st.info("No error logs available for analysis.")

    st.markdown("---")