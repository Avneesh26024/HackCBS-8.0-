# 🔮 Prism - AI-Driven Database Analytics

[![HackCBS 8.0](https://img.shields.io/badge/HackCBS-8.0-blue?style=flat-square)](https://github.com/Avneesh26024/HackCBS-8.0-)
[![Python](https://img.shields.io/badge/Python-3.9+-green?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal?style=flat-square)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue?style=flat-square)](https://reactjs.org/)

> Transform your database into a conversational AI assistant. Ask questions in natural language, get instant insights, visualizations, and exports—no SQL required.

**[🚀 Live Demo](https://db-agent-api-service-698063521469.asia-south1.run.app/docs)** | **[📖 Documentation](#getting-started)** | **[🎥 Demo Video](#)**

---

## 👥 Team MI-7

| Member | Role | GitHub |
|--------|------|--------|
| **Avneesh** | Backend &  AI Engineer| [@Avneesh26024](https://github.com/Avneesh26024) |
| **Jastej Singh** | Full Stack Engineer| [@JastejS28](https://github.com/JastejS28) |
| **Aditya Channa** | Database & AI Engineer | [@aditya](https://github.com/adityachanna) |
| **Harshit Chaudhry** | rontend & UI/UX  | [@adityasah](https://github.com/adityachanna) |
---

## 🎯 The Problem: Data Accessibility Crisis

Most organizations struggle with data accessibility:
- **Technical Barrier**: 85% of business users can't write SQL
- **Time Waste**: Data analysts spend 40% of time on repetitive queries
- **Limited Access**: Critical insights locked behind complex database schemas

**Our Solution**: Prism democratizes data access through conversational AI, RAG-powered schema understanding, and automated analytics.

---

## ✨ Key Features

### 🤖 Intelligent Query Understanding
- Natural language to SQL conversion using Gemini 2.5 Flash
- RAG-based schema comprehension with ChromaDB
- Multi-turn conversations with full context memory
- Support for complex joins, aggregations, and relationships

### 📊 Automated Analytics
- **Statistical Analysis**: Correlation, skewness, anomaly detection
- **Dynamic Visualizations**: Scatter plots, bar charts, line graphs, histograms
- **AI Vision Analysis**: Gemini Vision interprets plots and provides insights
- **Smart Export**: One-click PDF reports and Excel downloads

### 🗄️ Universal Database Support
- **SQL**: PostgreSQL, MySQL, SQLite, Supabase
- **Files**: CSV, Excel (XLSX/XLS)
- **Schema Auto-Discovery**: Automatic foreign key detection
- **Cloud-Ready**: Deployed on Google Cloud Run

### 🧠 Advanced RAG Architecture
- **Dual-Vector Store**: Separate collections for schemas and relationships
- **Semantic Search**: Gemini embeddings for context retrieval
- **Top-3 Results**: Precision-optimized RAG retrieval
- **Auto-Documentation**: Schema introspection and metadata extraction

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│               Frontend (React + Firebase)               │
│  • Firebase Auth  • Firestore (Chat)  • Hosted UI      │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS/REST API
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Backend (FastAPI on Cloud Run)                 │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │         LangGraph Agent Workflow (10 Nodes)       │  │
│  │  Intent → RAG → SQL Gen → Execute → Analyze      │  │
│  │  → Vision → Export → Response                     │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ ChromaDB │  │  DuckDB  │  │  GCS Storage │
│ (Vectors)│  │ (Query)  │  │   (Files)    │
└──────────┘  └──────────┘  └──────────────┘
```

**Workflow**: 
1. User query → Intent classification → RAG retrieval
2. SQL generation → Execution → Result analysis
3. Plot generation → Vision analysis → Export
4. Markdown response with embedded visualizations

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | Google Gemini 2.5 Flash, ChromaDB, LangChain, LangGraph |
| **Backend** | FastAPI, Python 3.9+, SQLAlchemy, DuckDB, Pandas |
| **Frontend** | React 18, Firebase (Auth/Firestore/Hosting), Tailwind CSS |
| **Cloud** | Google Cloud Run, Google Cloud Storage |
| **Databases** | PostgreSQL, MySQL, SQLite, CSV/Excel |
| **Visualization** | Matplotlib, FPDF, Excel Export |

---

## 🚀 Getting Started

### Prerequisites
```bash
python --version  # 3.9+
node --version    # 16+
```

### Backend Setup

1. **Clone Repository**
```bash
git clone https://github.com/Avneesh26024/HackCBS-8.0-.git
cd HackCBS-8.0-
```

2. **Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure Environment**
Create `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
Open_router_embedder_API_KEY=your_openrouter_key
GCS_BUCKET_NAME=your_gcs_bucket
```

4. **Run Server**
```bash
uvicorn api:app_api --host 0.0.0.0 --port 8000 --reload
```
API Docs: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to Frontend**
```bash
git clone https://github.com/JastejS28/HackCBS.git
cd HackCBS/frontend
npm install
```

2. **Configure Firebase**
Create `src/firebase-config.js` with your Firebase credentials

3. **Run Development Server**
```bash
npm start
```
App: `http://localhost:3000`

---

## 📡 API Endpoints

### `POST /upload_db`
Load database and build RAG system
```json
{
  "source": "postgresql://user:pass@host:port/db"
}
```

### `POST /chat`
Send conversational query
```json
{
  "messages": [
    {"type": "human", "content": "Show sales by region"}
  ]
}
```

### `POST /3d_generate`
Generate 3D schema visualization JSON
```json
{
  "source": "mysql://user:pass@host/db"
}
```

---

## 💡 Usage Examples

**Query**: "How many orders were placed last month?"  
**Response**: Executes SQL, returns count with formatted results

**Query**: "Plot revenue by product category"  
**Response**: Generates bar chart, uploads to GCS, provides visual analysis via Gemini Vision

**Query**: "Find anomalies in customer spending"  
**Response**: Runs statistical analysis, highlights outliers, generates report

**Query**: "Export all users to Excel"  
**Response**: Creates Excel file, uploads to cloud, provides download link

---

## 🎯 Project Structure
```
├── api.py                  # FastAPI endpoints
├── main.py                 # LangGraph workflow (10-node agent)
├── query_tool.py           # DataEngine (DuckDB + SQLAlchemy)
├── embedding_manager.py    # RAG system (ChromaDB)
├── image_result.py         # Plot/PDF/Excel generation
├── upload_to_uri.py        # GCS uploader with signed URLs
├── requirements.txt        # Python dependencies
└── db_vector_stores/       # ChromaDB persistent storage
```

---

## 🏆 HackCBS 8.0 Judging Criteria Alignment

| Criteria | Implementation |
|----------|----------------|
| **Technicality** | LangGraph state machine, dual-vector RAG, Gemini Vision integration, cloud deployment |
| **Originality** | Conversational database interface, AI-powered visual analysis, multi-format exports |
| **Practicality** | Production-ready API, 6 database types, deployed on Cloud Run, Firebase integration |
| **Design** | Clean React UI, Markdown responses, interactive charts, intuitive conversation flow |
| **WOW Factor** | "Talk to your database", AI explains plots, zero SQL knowledge required |

---

## 🌐 Deployment

### Backend (Google Cloud Run)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/prism-backend
gcloud run deploy prism-backend \
  --image gcr.io/PROJECT_ID/prism-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Frontend (Firebase Hosting)
```bash
npm run build
firebase deploy
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built with ❤️ for **HackCBS 8.0** by Team MI-7

Special thanks to:
- Google Cloud for infrastructure and Gemini
- HackCBS Team for thier support
- Major League Hacking for the opportunity

---

**[⬆ Back to Top](#-prism---ai-driven-database-analytics)**
