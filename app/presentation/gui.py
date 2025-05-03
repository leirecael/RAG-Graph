import streamlit as st
from logic.orchestrator import process_question
from data.log_reader import read_query_logs, read_error_logs
from data.logger import log_error
import pandas as pd
async def start_interface():
    st.set_page_config(page_title="Q&A System", layout="wide")
    page = st.sidebar.selectbox("Navegación", ["Consultas", "Historial", "Logs", "Estadísticas"])
    st.title("Sistema de preguntas y respuestas")
    
    if page == "Consultas":
        st.warning("Las preguntas se almacenarán para análisis de datos. No intruduzcas información personal.")
        if "history" not in st.session_state:
            st.session_state.history = []

        question = st.text_input("Introduce tu pregunta: ")


        response_placeholder = st.empty()

        if st.button("Preguntar"):
            if question:
                with st.spinner("Procesando..."):
                    try:
                        response = await process_question(question)
                        st.session_state.history.append({
                            "question": question,
                            "response": response
                        })

                        response_placeholder.write(response)
                    except Exception as e:
                        response_placeholder.error("Ocurrió un error inesperado")
                        log_error("GUIError", {
                            "error": str(e), 
                        })
                    
            else:
                st.warning("Por favor, introduce una pregunta.")

    elif page == "Historial":
        st.subheader("Historial de consultas")
        if st.session_state.get("history"):
            st.markdown("---")

            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Pregunta #{len(st.session_state.history) - idx}: {entry['question']}"):
                    st.markdown(f"**Respuesta:**\n> {entry['response']}")

            if st.button("Borrar historial"):
                st.session_state.history = []
                st.success("Historial borrado correctamente.")
        else:
            st.info("Todavía no has hecho ninguna pregunta.")

    elif page == "Logs":
        st.subheader("Logs")
        log_type = st.radio("Selecciona tipo de log", ["Consultas", "Errores"])

        if log_type == "Consultas":
            logs = read_query_logs()
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df)
            else:
                st.info("No hay logs de consultas.")

        else:
            logs = read_error_logs()
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df)
            else:
                st.info("No hay logs de errores.")

    elif page == "Estadísticas":
        st.subheader("Estadísticas del sistema")

        query_logs = read_query_logs()
        error_logs = read_error_logs()

        if query_logs:
            df = pd.DataFrame(query_logs)

            # Costes totales
            total_cost_queries = df["total_cost"].sum()

            # Procesar errores con coste
            if error_logs:
                error_df = pd.DataFrame(error_logs)
                for col in ["cost_gpt", "cost_embed", "cost_agent", "total_cost"]:
                    error_df[col] = pd.to_numeric(error_df.get(col, 0), errors='coerce')
                total_cost_errors = error_df["total_cost"].sum()
            else:
                total_cost_errors = 0

            total_cost_global = total_cost_queries + total_cost_errors

            # Conversión de columnas numéricas
            for col in ["time_elapsed", "cost_gpt", "cost_embed", "cost_agent", "total_cost"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            st.markdown("### Métricas generales")

            st.markdown("### Coste total acumulado")

            col1, col2, col3 = st.columns(3)
            col1.metric("Coste total de consultas", f"{total_cost_queries:.4f}€")
            col2.metric("Coste total de errores", f"{total_cost_errors:.4f}€")
            col3.metric("Coste total global", f"{total_cost_global:.4f}€")

            col1, col2, col3 = st.columns(3)
            col1.metric("Tiempo medio de respuesta", f"{df['time_elapsed'].mean():.2f} s")
            col2.metric("Coste total medio", f"{df['total_cost'].mean():.4f}€")
            col3.metric("Consultas registradas", len(df))

            st.markdown("### Costes por componente")
            st.bar_chart(df[["cost_gpt", "cost_embed", "cost_agent"]].mean())

            st.markdown("### Distribución del coste total")
            st.line_chart(df["total_cost"])

            st.markdown("### Top 5 preguntas más costosas")
            top_cost = df.sort_values("total_cost", ascending=False).head(5)[["question", "total_cost"]]
            st.table(top_cost)

        else:
            st.info("No hay logs de consultas suficientes para mostrar estadísticas.")

        st.markdown("---")
        st.markdown("### Errores más comunes")

        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_counts = error_df["error_type"].value_counts().reset_index()
            error_counts.columns = ["Tipo de error", "Frecuencia"]
            st.bar_chart(error_counts.set_index("Tipo de error"))
        else:
            st.info("No hay logs de errores para analizar.")

    st.markdown("---")