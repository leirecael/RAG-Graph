# RAG-Graph

## Prerequisites

- **Python**: v3.12.9
- **Credentials**:
  - Neo4j database up and running in local/remote
  - Neo4j database credentials
  - OpenAI API key

Create a `.env` file in the app/config/.env path of the project with the following content:

```env
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
OPENAI_API_KEY=your_openai_api_key
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/leirecael/RAG-Graph.git
cd RAG-Graph
```

### 2. Create virtual environment and access it(optional)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install libraries and dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

Windows:

```bash
streamlit run app\main.py
```

Linux/macOS:

```bash
streamlit run app/main.py
```

Once the app is running, go to your web browser and visit:

```text
http://localhost:8501/
```
