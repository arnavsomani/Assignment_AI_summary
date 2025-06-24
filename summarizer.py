import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List
import logging
import os

# model from hugging face

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# facebook/bart-large-cnn

class TextSummarizer:
    """
    A text summarizer using Hugging Face transformers
    """
    
    def __init__(self, model_name = model):
        """
        Initialize the summarizer with a pre-trained model
        
        Args:
            model_name: The name of the model to use for summarization
        """
        self.model_name = model_name
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            # Use CPU for compatibility, GPU if available
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=tokenizer,
                device=device,
                framework="pt"
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a smaller model if the main model fails
            try:
                logger.info("Trying fallback model: sshleifer/distilbart-cnn-12-6")
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1  # Force CPU for fallback
                )
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise Exception("Unable to load any summarization model")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        
        return text
    
    def chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        """
        Split text into chunks if it's too long for the model
        
        Args:
            text: Input text to chunk
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize(self, text: str) -> List[str]:
        """
        Generate a three-point summary from the input text
        
        Args:
            text: Input text to summarize
            
        Returns:
            List of three summary points
        """
        if not self.summarizer:
            raise Exception("Summarizer model not loaded")
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Check if text is long enough
        word_count = len(text.split())
        if word_count < 500:
            raise ValueError("Text must be at least 500 words long")
        
        try:
            # Chunk text if necessary
            chunks = self.chunk_text(text, max_length=1000)
            
            summaries = []
            for chunk in chunks:
                # Generate summary for each chunk
                summary = self.summarizer(
                    chunk,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    truncation=True
                )
                summaries.append(summary[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_summary = ' '.join(summaries)
                # Summarize the combined summary to get final result
                final_summary = self.summarizer(
                    combined_summary,
                    max_length=200,
                    min_length=100,
                    do_sample=False,
                    truncation=True
                )
                summary_text = final_summary[0]['summary_text']
            else:
                summary_text = summaries[0]
            
            # Convert to three bullet points
            bullet_points = self._extract_bullet_points(summary_text)
            
            return bullet_points
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise Exception(f"Summarization failed: {str(e)}")
    
    def _extract_bullet_points(self, summary: str) -> List[str]:
        """
        Convert summary text into three distinct bullet points
        
        Args:
            summary: Generated summary text
            
        Returns:
            List of three bullet points
        """
        # Split by sentences
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have exactly 3 or fewer sentences, return them
        if len(sentences) <= 3:
            # Pad with additional analysis if needed
            while len(sentences) < 3:
                sentences.append("Additional analysis and context would provide more comprehensive insights.")
            return sentences[:3]
        
        # If more than 3 sentences, intelligently select the most important ones
        # For now, take first, middle, and last sentences
        if len(sentences) > 3:
            selected = [
                sentences[0],  # Introduction/main point
                sentences[len(sentences)//2],  # Middle point
                sentences[-1]  # Conclusion
            ]
            return selected
        
        return sentences[:3]

# Test function
def test_summarizer():
    """Test function for the summarizer"""
    sample_text = """
    Artificial intelligence has revolutionized the way we approach problem-solving and data analysis across numerous industries. From healthcare to finance, AI technologies are being implemented to improve efficiency, accuracy, and decision-making processes. Machine learning algorithms can now process vast amounts of data in seconds, identifying patterns and insights that would take humans years to discover. Deep learning networks, inspired by the human brain's neural structure, have enabled computers to perform complex tasks such as image recognition, natural language processing, and predictive analytics with remarkable precision.

    The healthcare industry has particularly benefited from AI integration, with applications ranging from diagnostic imaging to drug discovery. AI-powered systems can analyze medical scans with greater accuracy than human radiologists in many cases, leading to earlier detection of diseases and improved patient outcomes. Additionally, AI algorithms are being used to develop personalized treatment plans based on individual patient data, genetic information, and medical history.

    However, the rapid advancement of AI also brings challenges and ethical considerations. Issues such as job displacement, privacy concerns, and algorithmic bias need to be addressed as AI becomes more prevalent in society. The development of responsible AI practices and regulatory frameworks is crucial to ensure that these technologies benefit humanity while minimizing potential risks and negative consequences.
    """
    
    summarizer = TextSummarizer()
    try:
        summary = summarizer.summarize(sample_text)
        print("Summary points:")
        for i, point in enumerate(summary, 1):
            print(f"{i}. {point}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_summarizer()