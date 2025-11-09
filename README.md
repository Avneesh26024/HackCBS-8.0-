# AI-Driven DB RAG & Analytics

An intelligent data assistant that understands your database schema and answers natural-language queries. Generate live visualizations, detect anomalies, deliver insights, and export results to PDF/Excel — all powered by custom RAG (Retrieval Augmented Generation).


## Live Demo
**[Try it now → Prism]()**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Judging Criteria Alignment](#judging-criteria-alignment)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project transforms how users interact with databases by providing a conversational AI interface that:
- Understands database schemas through intelligent embeddings
- Generates SQL queries from natural language
- Creates visualizations and performs statistical analysis
- Exports results in multiple formats (PDF, Excel)
- Maintains conversation context for follow-up questions

**Problem Statement**: Create an AI-powered data assistant that understands a user's database schema and answers natural-language queries. Generate live visualizations, detect anomalies, deliver insights, and export results to PDF/Excel — powered by custom RAG.

## Features

### Intelligent Query Understanding
- Natural language to SQL conversion
- Context-aware query generation using RAG
- Support for complex joins and aggregations
- Schema relationship understanding
- Multi-turn conversation with memory

### Visual Analytics
- Automatic plot generation (scatter, bar, line, histogram)
- Statistical analysis (correlation, skewness, anomaly detection)
- AI-powered visual interpretation using Gemini Vision
- Interactive chart rendering
- Real-time data visualization

### Multi-Format Export
- PDF reports with formatted tables
- Excel spreadsheets with preserved data types
- Cloud-hosted files with temporary signed URLs
- Automatic file management
- One-click download links

### Universal Database Support
- **PostgreSQL** - Full support with schema introspection
- **MySQL** - Native driver with system table filtering
- **SQLite** - Lightweight embedded database support
- **CSV/Excel** - Direct file analysis
- **Supabase** - Cloud database integration

### Advanced RAG System
- Dual-collection vector store (schemas + relationships)
- ChromaDB for efficient similarity search
- Gemini embeddings for semantic understanding
- Context-aware retrieval with 3 top results
- Automatic schema documentation

### Conversational Interface
- Multi-turn conversations with full history
- Follow-up question handling
- Intent classification (new query vs. history answer)
- Natural language responses
- Markdown-formatted outputs

## Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend (React + Firebase)                     │
│                                                                     │
│    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│    │    Firebase     │  │    Firebase     │  │     Chat UI     │    │
│    │      Auth       │  │   Firestore     │  │   Component     │    │
│    │  (User Login)   │  │ (Chat History)  │  │  (Interface)    │    │
│    └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ REST API (HTTPS)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Backend (FastAPI on Google Cloud Run)                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   LangGraph Agent Workflow                  │    │
│  │                                                             │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │    │
│  │  │  Intent  │ → │   RAG    │ → │   SQL    │ → │ Analysis │  │    │
│  │  │ Classify │   │ Retrieval│   │   Gen    │   │ & Plot   │  │    │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │    │
│  │                                                             │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │    │
│  │  │  Router  │ → │  Vision  │ → │  Export  │ → │ Response │  │    │
│  │  │  Logic   │   │ Analysis │   │  Check   │   │ Generate │  │    │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │    ChromaDB       │ │   Data Engine     │ │  Google Cloud     │
    │ (Vector Store)    │ │  (Query Engine)   │ │    Storage        │
    │                   │ │                   │ │                   │
    │ • Table Schemas   │ │ • DuckDB          │ │ • Plot Images     │
    │ • Relationships   │ │ • SQLAlchemy      │ │ • PDF Reports     │
    │ • Embeddings      │ │ • Multi-DB        │ │ • Excel Files     │
    │                   │ │   Support         │ │ • Signed URLs     │
    └───────────────────┘ └───────────────────┘ └───────────────────┘

```

### Workflow Explanation
1. **Intent Classification**: Determines if query needs database access or can be answered from chat history
2. **RAG Retrieval**: Fetches relevant schema and relationship information from ChromaDB
3. **SQL Generation**: Uses Gemini 2.5 Flash to create optimized SQL queries
4. **SQL Execution**: Runs query through DuckDB/SQLAlchemy hybrid engine
5. **Analysis Router**: Decides if visualization/statistics are needed
6. **Code Generation**: Creates Python analysis code dynamically
7. **Execution & Upload**: Runs code, generates files, uploads to GCS
8. **Visual Analysis**: Gemini Vision analyzes plots and provides insights
9. **Export Check**: Handles PDF/Excel export if requested
10. **Response Generation**: Creates formatted Markdown response with all results

## Tech Stack

### Backend Services
| Technology | Purpose |
|------------|---------|
| **Google Cloud Run** | Serverless deployment |
| **Google Gemini AI** | LLM & Vision analysis (2.5 Flash) |
| **ChromaDB** | Vector database |
| **Google Cloud Storage** | File hosting with signed URLs |

### Backend Framework
- **FastAPI** - Modern web framework
- **LangChain** - LLM application framework
- **LangGraph** - State machine orchestration
- **DuckDB** - In-memory analytics engine
- **SQLAlchemy** - SQL toolkit and ORM
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **FPDF** - PDF generation

### Frontend Services
| Technology | Purpose |
|------------|---------|
| **Firebase Auth** | User authentication |
| **Firebase Firestore** | Chat history storage |
| **Firebase Hosting** | Static site hosting |

### Frontend Framework
- **React 18+** - UI library
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering

### Supported Databases
- PostgreSQL
- MySQL
- SQLite
- CSV files
- Excel files
- Supabase

## Getting Started

### Prerequisites
```bash
python --version  # 3.9+
node --version    # 16+
gcloud --version
```

### Backend Setup

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/db-rag-analytics.git
cd db-rag-analytics
```

#### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Configure Environment
Create `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
Open_router_embedder_API_KEY=your_openrouter_key
GCS_BUCKET_NAME=hackcbs_generate_uri
```

#### 4. Run Server
```bash
uvicorn api:app_api --host 0.0.0.0 --port 8000 --reload
```
API available at: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### Frontend Setup

#### 1. Navigate to Frontend
```bash
cd frontend
npm install
```

#### 2. Configure Firebase
Create `src/firebase-config.js`:
```javascript
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

export const firebaseConfig = {
  apiKey: "your_api_key",
  authDomain: "your_project.firebaseapp.com",
  projectId: "your_project_id",
  storageBucket: "your_project.appspot.com",
  messagingSenderId: "123456789",
  appId: "your_app_id"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
```

#### 3. Run Development Server
```bash
npm start
```
Frontend available at: `http://localhost:3000`

## Project Structure
```
db-rag-analytics/
├── api.py                      # FastAPI endpoints
├── main.py                     # LangGraph workflow
├── query_tool.py              # DataEngine
├── embedding_manager.py       # RAG system
├── image_result.py            # Export utilities
├── upload_to_uri.py           # GCS uploader
├── utils.py                   # Embeddings
├── requirements.txt           # Dependencies
├── db_vector_stores/          # ChromaDB data
└── results/                   # Generated files


```

## API Endpoints

### POST /upload_db
Upload database source and build RAG system.

**Request:**
```json
{
  "source": "postgresql://user:pass@host:port/db"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Data source loaded and vector stores built.",
  "db_structure": {
    "tables": [
      {"name": "customers", "columns": 5, "rows": 1500, "foreign_keys": 0}
    ],
    "relationships": []
  }
}
```

### POST /chat
Send conversational query.

**Request:**
```json
{
  "messages": [
    {"type": "human", "content": "How many users are there?"}
  ]
}
```

**Response:**
```json
{
  "messages": [
    {"type": "human", "content": "How many users are there?"},
    {"type": "ai", "content": "## Result\n\nThere are 1,500 users.\n\n**SQL:**\n```sql\nSELECT COUNT(*) FROM users\n```"}
  ]
}
```

### POST /3d_generate
Generate 3D schema visualization JSON.

**Request:**
```json
{
  "source": "mysql://user:pass@host:port/db"
}
```

**Response:**
```json
{
  "schema_name": "ecommerce",
  "nodes": [{"id": "table_customers", "name": "customers", "attributes": []}],
  "edges": []
}
```

## Usage Examples

### Example 1: Basic Query
```
User: "How many orders were placed last month?"
AI: 245 orders were placed last month.
```

### Example 2: Visualization
```
User: "Plot sales by category"
AI: [Generates bar chart with breakdown]
```

### Example 3: Anomaly Detection
```
User: "Find anomalies in order amounts"
AI: Detected 3 anomalies: Order #1523 ($15,000), Order #2891 ($0)...
```

### Example 4: Export
```
User: "Export all users to Excel"
AI: [Provides download link to Excel file]
```

## Deployment

### Backend (Google Cloud Run)
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/db-rag-backend
gcloud run deploy db-rag-backend \
  --image gcr.io/PROJECT_ID/db-rag-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=xxx,Open_router_embedder_API_KEY=xxx
```

### Frontend (Firebase Hosting)
```bash
npm run build
firebase login
firebase init hosting
firebase deploy
```

## Judging Criteria Alignment

### Technicality (✓)
- Fully functional RAG system with dual vector stores
- Complex LangGraph state machine with 10+ nodes
- Integration of multiple AI models (Gemini LLM + Vision)
- Real-time SQL generation and execution
- Advanced file handling with cloud storage

### Originality (✓)
- Novel dual-collection RAG approach (schemas + relationships)
- AI-powered visual analysis using Gemini Vision
- Conversational database interaction paradigm
- Automatic intent classification and routing

### Practicality (✓)
- Supports 5+ database types out-of-the-box
- Production-ready FastAPI backend
- Firebase-integrated React frontend
- Real PDF/Excel export functionality
- Deployed on Google Cloud Run

### Design (✓)
- Clean, modern React UI
- Intuitive conversation flow
- Professional markdown-formatted responses
- Visual hierarchy with charts and tables

### WOW Factor (✓)
- "Talk to your database" experience
- AI explains visualizations in natural language
- Zero SQL knowledge required
- One-click deployment to cloud

---

**Built with ❤️ for HackCBS 2024**

*Transform your database into a conversational AI assistant*
