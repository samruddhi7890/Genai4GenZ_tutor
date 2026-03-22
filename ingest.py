import PyPDF2
import os
import json
import ast
import re
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

@st.cache_data(show_spinner=False)
def process_textbook(uploaded_file):
    try:
        # 1. READ THE PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # 2. BUILD THE FAISS DATABASE (Local)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")
        
        # 3. GENERATE JSON DATA VIA GROQ (Cloud)
        toc_text = full_text[:15000] 
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
        Analyze the following text from the beginning of a textbook. 
        Your ONLY job is to output a strictly valid JSON object representing the book's structure.
        
        INSTRUCTIONS:
        1. Extract the FIRST 8 main chapters.
        2. Format titles with Roman Numerals (I, II, III...).
        3. MAKE THE SUMMARIES LONG. Each chapter summary must be a highly detailed, comprehensive paragraph of at least 5 to 7 sentences (approximately 80-100 words). Since you are only seeing the Table of Contents, you MUST use your extensive educational knowledge to fill out these summaries with core concepts, historical context, and key takeaways related to the chapter's title.
        4. Include a 4-step 'roadmap'.
        5. Include 5 'exam_questions'.
        
        CRITICAL FAIL-SAFE: If you cannot find a clear Table of Contents, infer the topics. Output ONLY valid JSON.
        
        JSON STRUCTURE:
        {{
            "chapters": {{ "I. Chapter Name": ["This chapter provides an in-depth analysis of the topic. It begins by exploring the fundamental concepts and historical background that set the stage. Students will learn about the key figures, primary mechanisms, and underlying theories involved. Furthermore, the text breaks down complex scenarios into understandable frameworks. Finally, it highlights the real-world applications and long-term consequences of these events, ensuring a comprehensive grasp of the material."] }},
            "roadmap": ["Step 1", "Step 2", "Step 3", "Step 4"],
            "exam_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]
        }}
        
        Text: {toc_text}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a rigid data extraction API. You output raw, valid JSON and nothing else. Never use markdown. Never apologize."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=4000  # 🚨 THE FIX: Gives the AI enough room to write long summaries without getting cut off
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # 🚨 DEBUGGING
        print("\n--- RAW AI RESPONSE ---")
        print(raw_response)
        print("-----------------------\n")
        
        # 🚨 AGGRESSIVE CLEANUP
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json_string = raw_response[start_idx:end_idx+1]
            
            clean_json_string = re.sub(r',\s*}', '}', clean_json_string)
            clean_json_string = re.sub(r',\s*\]', ']', clean_json_string)
            
            try:
                return json.loads(clean_json_string)
            except json.JSONDecodeError:
                import ast
                return ast.literal_eval(clean_json_string)
        else:
            raise ValueError(f"No brackets found. The AI completely ignored the instructions. Raw output: {raw_response[:100]}...")

    except Exception as e:
        print(f"Groq Processing Error: {e}")
        return {"Error Processing Book": [f"Details: {str(e)}", "The AI generated poorly formatted data. Check your terminal for the raw output."]}