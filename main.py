
import streamlit as st
import sys
import os
from summarizer import TextSummarizer
import time


# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Initialize summarizer
@st.cache_resource
def load_summarizer():
    """Load and cache the summarizer model"""
    return TextSummarizer()

def main():
    st.title("🤖 AI-Enabled Text Summarizer")
    st.markdown("---")
    
    st.markdown("""
    ### Instructions:
    - Enter or paste text with **at least 500 words**
    - Click 'Generate Summary' to get a three-bullet-point summary
    - The app uses a pre-trained AI model for summarization
    """)
    
    # Text input area
    text_input = st.text_area(
        "Enter your text here:",
        height=300,
        placeholder="Paste your long-form text here (minimum 500 words)..."
    )
    
    # Word count display
    word_count = len(text_input.split()) if text_input else 0
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if word_count < 500 and word_count > 0:
            st.warning(f"Word count: {word_count}/500 - Please enter at least 500 words")
        elif word_count >= 500:
            st.success(f"Word count: {word_count} ✓")
    
    with col2:
        generate_button = st.button("🚀 Generate Summary", type="primary")
    
    # Process summarization
    if generate_button:
        if word_count < 500:
            st.error("❌ Please enter at least 500 words before generating a summary.")
        else:
            try:
                # Load summarizer
                summarizer = load_summarizer()
                
                # Show progress
                with st.spinner("🔄 Generating summary... This may take a moment."):
                    summary_points = summarizer.summarize(text_input)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Three-Point Summary:")
                
                for i, point in enumerate(summary_points, 1):
                    st.markdown(f"**{i}.** {point}")
                
                # Success message
                st.success("✅ Summary generated successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Try with different text or restart the app.")

if __name__ == "__main__":
    main()