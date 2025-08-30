# AI-Powered Personal Finance Assistant

## Overview

This project is an AI-powered personal finance assistant built with Streamlit that helps users analyze their financial transactions through natural language queries. The system automatically categorizes expenses using transformer models, enables semantic search over transaction history, and provides intelligent insights through conversational AI. Users can upload bank statements in CSV or Excel format and interact with their financial data using plain English questions to get summaries, trends, and spending analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application serving as the main user interface
- **Interactive Dashboard**: Real-time data visualization with Plotly charts and graphs
- **File Upload Interface**: Supports CSV and Excel file formats with validation
- **Chat Interface**: Conversational AI interface for natural language queries

### Backend Architecture
- **Modular Component Design**: Separated into distinct classes for different functionalities
  - `DataProcessor`: Handles file loading and data preprocessing
  - `ExpenseCategorizer`: Automatic expense categorization using transformer models
  - `VectorStore`: Manages vector embeddings and semantic search
  - `ConversationHandler`: Processes natural language queries and generates responses
  - `FinanceVisualizer`: Creates interactive financial visualizations

### Data Processing Pipeline
- **Data Ingestion**: Automatic detection and parsing of CSV/Excel files
- **Data Cleaning**: Column mapping, standardization, and validation
- **Expense Categorization**: Zero-shot classification using transformer models with fallback keyword matching
- **Vector Embedding**: Transaction descriptions converted to embeddings for semantic search

### AI and ML Components
- **Transformer Models**: HuggingFace transformers for zero-shot expense classification
- **OpenAI Integration**: GPT-5 model for conversational AI and query processing
- **Vector Search**: ChromaDB for storing and retrieving transaction embeddings
- **Semantic Search**: Context-aware transaction retrieval based on user queries

### Data Storage
- **In-Memory Vector Database**: ChromaDB configured for in-memory storage
- **Session State Management**: Streamlit session state for maintaining user data and conversation history
- **Transaction Memory**: Pandas DataFrames for structured data manipulation

### Visualization System
- **Interactive Charts**: Plotly-based visualizations for spending trends and category breakdowns
- **Real-time Updates**: Dynamic chart generation based on data filters and user interactions
- **Multi-format Support**: Pie charts, line graphs, and trend analysis

## External Dependencies

### AI and ML Services
- **OpenAI API**: GPT-5 model for conversational AI and natural language processing
- **HuggingFace Transformers**: Pre-trained models for expense categorization
- **ChromaDB**: Vector database for semantic search and embeddings storage

### Python Libraries
- **Streamlit**: Web application framework and user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive data visualization and charting

### Data Processing
- **File Format Support**: Native CSV and Excel file processing
- **Text Processing**: Regular expressions and string manipulation for data cleaning
- **Date/Time Handling**: DateTime operations for temporal analysis

### Infrastructure
- **Environment Variables**: OpenAI API key configuration
- **Memory Management**: In-memory data storage for session persistence
- **Error Handling**: Comprehensive exception handling and user feedback