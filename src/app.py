import streamlit as st
import pandas as pd
import plotly.express as px
from models.llm_handler import LLMHandler
from models.image_analyzer import DeepImageAnalyzer
import torch
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Robot Manual Analysis System",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_models():
    """Initialize models with caching for better performance."""
    llm = LLMHandler(use_streamlit=True)
    image_analyzer = DeepImageAnalyzer()
    return llm, image_analyzer

def main():
    st.title("🤖 Advanced Robot Manual Analysis System")
    st.markdown("""
    This system uses state-of-the-art AI models to analyze robot manuals:
    - Text analysis using ensemble of RoBERTa and DeBERTa models
    - Image analysis using Vision Transformers
    - Explainable AI features
    """)
    
    llm, image_analyzer = initialize_models()
    
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "📊 Results Dashboard"])
    
    with tab1:
        st.header("Text Analysis")
        text_input = st.text_area(
            "Enter text from robot manual:",
            height=200,
            help="Paste a section of text from a robot manual for analysis"
        )
        
        if st.button("Analyze Text", key="analyze_text"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    # Get prediction and explanation
                    prediction = llm.ensemble_predict([text_input])[0]
                    explanation = llm.explain_prediction(text_input)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Prediction")
                        st.info(f"Category: {prediction['prediction']}")
                        st.progress(prediction['confidence'])
                        st.write(f"Confidence: {prediction['confidence']:.2%}")
                        if prediction['model_agreements']:
                            st.success("✅ All models agree on this prediction")
                        else:
                            st.warning("⚠️ Models have different predictions")
                            
                    with col2:
                        st.subheader("Explanation")
                        explanation_df = pd.DataFrame(
                            explanation['explanation'],
                            columns=['Text', 'Impact']
                        )
                        fig = px.bar(
                            explanation_df,
                            x='Impact',
                            y='Text',
                            orientation='h',
                            title='Feature Importance'
                        )
                        st.plotly_chart(fig)
            else:
                st.warning("Please enter some text to analyze")
    
    with tab2:
        st.header("Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload an image from robot manual",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image", key="analyze_image"):
                with st.spinner("Analyzing image..."):
                    analysis = image_analyzer.analyze_image(image)
                    
                    st.subheader("Image Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"Content Type: {analysis.content_type}")
                        st.progress(analysis.confidence)
                        st.write(f"Confidence: {analysis.confidence:.2%}")
                        
                    with col2:
                        st.write("Technical Details:")
                        for key, value in analysis.technical_details.items():
                            st.write(f"- {key}: {value}")
                            
                    st.write("Description:")
                    st.write(analysis.description)
    
    with tab3:
        st.header("Results Dashboard")
        st.info("Upload a CSV file with analysis results to visualize metrics")
        
        results_file = st.file_uploader(
            "Upload results CSV",
            type=['csv'],
            key="results_upload"
        )
        
        if results_file:
            results_df = pd.read_csv(results_file)
            
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(
                    results_df,
                    x='epoch',
                    y=['train_loss', 'val_loss'],
                    title='Training Progress'
                )
                st.plotly_chart(fig1)
                
            with col2:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                fig2 = px.bar(
                    results_df[metrics].iloc[-1:].melt(),
                    x='variable',
                    y='value',
                    title='Final Model Performance'
                )
                st.plotly_chart(fig2)

if __name__ == "__main__":
    main() 