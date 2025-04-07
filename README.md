# Text Mining Resume Dashboard

A modern web application for text mining and analysis, featuring a Next.js frontend and FastAPI backend. This dashboard provides powerful tools for text analysis, resume parsing, and job description evaluation.

[Demo Video](https://drive.google.com/file/d/1bdR-y6Lux8t11ojkR928-q4co2MNAYZX/view?usp=sharing)

## 🚀 Features

- **Modern UI**: Built with Next.js 15 and Tailwind CSS
- **Text Analysis**: Advanced text mining capabilities
- **Resume Parser**: Intelligent resume analysis and evaluation
- **Job Description Analyzer**: Comprehensive JD evaluation tools
- **Interactive Dashboard**: Real-time data visualization
- **Responsive Design**: Works seamlessly across all devices

## 🛠️ Tech Stack

### Frontend
- Next.js 15
- React 19
- Tailwind CSS
- Radgit push -u origin mainix UI Components
- Recharts for data visualization
- TypeScript

### Backend
- FastAPI
- LangChain
- Sentence Transformers
- PyPDF2
- scikit-learn
- NLTK
- PyTorch

## 📋 Prerequisites

- Node.js (Latest LTS version)
- Python 3.8+
- pnpm (recommended) or npm
- Virtual environment for Python

## 🚀 Getting Started

### Backend Setup

1. Navigate to the server directory:
   ```bash
   cd server
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the client directory:
   ```bash
   cd client
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Start the development server:
   ```bash
   pnpm dev
   ```

The application will be available at `http://localhost:3000`

## 📁 Project Structure

```
dashboard/
├── client/                 # Frontend Next.js application
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   ├── hooks/            # Custom React hooks
│   ├── lib/              # Utility functions
│   ├── public/           # Static assets
│   └── styles/           # CSS styles
│
├── server/                # Backend FastAPI application
│   ├── main.py           # FastAPI application entry point
│   ├── evaluation_helper.py  # Text evaluation utilities
│   ├── resume_helper.py  # Resume parsing utilities
│   └── jd_helper.py      # Job description analysis
│
└── assets/               # Project assets
    └── demo.mp4          # Demo video
```

## 🔧 Configuration

The application uses various configuration files:

- `client/next.config.mjs`: Next.js configuration
- `client/tailwind.config.js`: Tailwind CSS configuration
- `client/tsconfig.json`: TypeScript configuration
- `server/main.py`: FastAPI server configuration

## 📝 API Documentation

The backend API documentation is available at `http://localhost:8000/docs` when the server is running.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

