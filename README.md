### AI-Enabled Text Summarizer

A lightweight Python application that generates three-bullet-point summaries from long-form text using Hugging Face transformer models.

### Architecture Components:

1. **Frontend (Streamlit)**  
   - Handles user input/output with real-time word counting
   - Manages application state and UI components
   - Displays summary results and error messages

2. **Backend (TextSummarizer)**  
   - **Text Preprocessor**: Cleans and normalizes input text
   - **Chunking Engine**: Splits long texts into model-digestible segments
   - **Model Manager**:  
     - Primary model: `Falconsai/text_summarization`  
     - Fallback: `sshleifer/distilbart-cnn-12-6` (CPU-optimized)
   - **Summary Post-processor**: Formats output into three bullet points

3. **AI Models**  
   - Transformer-based seq2seq models fine-tuned for summarization
   - Automatic device detection (GPU/CPU)
   - Model caching for faster subsequent runs

## Design Philosophy

### Core Principles:
1. **Offline-First Approach**  
   - All models run locally after initial download
   - No API keys or external dependencies required
   - Automatic fallback to CPU-optimized model

2. **Robust Text Handling**  
   - Intelligent text chunking for long documents
   - Multi-stage summarization for cohesive outputs
   - Sentence extraction heuristics for consistent 3-point format

3. **Error Resilience**  
   - Graceful degradation with model fallbacks
   - Comprehensive input validation
   - Detailed error logging for troubleshooting

4. **User Experience Focus**  
   - Real-time word count feedback
   - Progress indicators during processing
   - Clear success/error states

## Setup Instructions

### 1. Installation Steps
git clone https://github.com/your-repo/ai-summarizer-app

cd ai-summarizer-app

### Create virtual environment and activate
python -m venv venv
### Mac: source venv/bin/activate 
### Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Download models (automatic on first run)

streamlit run main.py

### Wait for model download completion (2-5 minutes)
