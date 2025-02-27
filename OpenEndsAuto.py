import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import numpy as np
import os, json, time, re, ast
import openai
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

# ---------------------------
# Page and Session Configurations
# ---------------------------
st.set_page_config(
    page_title="Survey Coding Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS for Modern UI
# ---------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .card {
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Button enhancements */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #4CAF50, #45a049);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .card {
            background: #2c3e50;
            color: #ecf0f1;
        }
        h2 {
            color: #ecf0f1;
            border-color: #4CAF50;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Session State Initialization
# ---------------------------
session_defaults = {
    'api_key': "",
    'client': None,
    'questions': None,
    'verbatims': None,
    'codeframes': {},
    'coded_data': {},
    'topic_model': {},
    'code_counter': 1
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------
# Helper Functions
# ---------------------------
def call_openai_api(prompt, model, max_tokens, temperature, stop_sequences=None, api_key=None):
    if api_key is None:
        api_key = st.session_state.get("api_key", "")
    openai.api_key = api_key
    try:
        st.session_state.last_api_request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_sequences
        }
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences
        )
        return response
    except Exception as e:
        st.error(f"API Call Error: {str(e)}")
        raise

def init_openai_client(api_key):
    try:
        test_prompt = "Hello OpenAI!"
        model_name = "gpt-4o-mini"
        with st.spinner("Testing API connection..."):
            _ = call_openai_api(
                prompt=test_prompt,
                model=model_name,
                max_tokens=50,
                temperature=0.0
            )
        st.session_state.client = True
        st.session_state.current_model = model_name
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        if hasattr(st.session_state, 'last_api_request'):
            with st.expander("Debug Request Information"):
                st.json(st.session_state.last_api_request)
        return False

def validate_json_response(response_text):
    try:
        cleaned = re.sub(r'^```json\s*|\s*```$', '', response_text)
        cleaned = re.sub(r'(?<={|,)\s*([a-zA-Z_]+)\s*:', r'"\1":', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.error(f"JSON validation failed: {str(e)}\nRaw response: {response_text}")
        return None

def update_codeframe(global_codeframe, batch_codeframe):
    for code_name, details in batch_codeframe.items():
        exists = any(v.get("code_name") == code_name for v in global_codeframe.values())
        if not exists:
            code_number = st.session_state.code_counter
            st.session_state.code_counter += 1
            global_codeframe[code_number] = {
                "code_name": code_name,
                "description": details.get("description", ""),
                "keywords": details.get("keywords", [])
            }
    return global_codeframe

def display_codeframe(codeframe):
    if not codeframe:
        st.warning("No codeframe generated yet")
        return
    
    # Add error code with proper keyword formatting
    error_code = {
        "Code Number": 999,
        "Code Name": "Processing Error",
        "Description": "Failed to code this response",
        "Keywords": ""  # Changed from empty list to empty string
    }
    
    df = pd.DataFrame([
        error_code,
        *[{
            "Code Number": code_num,
            "Code Name": details.get("code_name", ""),
            "Description": details.get("description", ""),
            "Keywords": ", ".join(details.get("keywords", []))  # Ensure this is always a string
        } for code_num, details in codeframe.items()]
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)

def generate_codeframe_batch(responses, question_text, num_codes=10):
    responses_text = "\n".join(responses)
    prompt = f"""[Question Text]:
{question_text}

[Verbatims]:
{responses_text}

You are a skilled qualitative researcher tasked with creating a robust code frame to analyze the above survey responses.
Your code frame must:
- Include a hierarchical structure (Nets, Sub Nets, and Sub Sub Nets as needed).
- Include a Net titled “Don’t know/Nothing/No answer” with these fixed codes:
    - Code Number: 99, Code Name: Don’t know
    - Code Number: 97, Code Name: Nothing
    - Code Number: 98, Code Name: No answer
- For all other codes, assign each a unique numeric Code Number (starting at 1, whole numbers only; do not assign numbers to Nets/Sub Nets).
- Ensure codes at the same level are mutually exclusive and exhaustive.
Output only the final code frame as a JSON object.
IMPORTANT: Output ONLY the valid JSON object with no extra text.
For example:
{{"Food Quality": {{"description": "Responses related to taste, freshness, and appearance", "keywords": ["taste", "fresh", "appearance"]}}, "Service": {{"description": "Responses about customer service", "keywords": ["service", "staff"]}}}}
Assistant:"""
    with st.spinner("Generating codeframe..."):
        try:
            model_name = st.session_state.get("current_model", "gpt-4o-mini")
            api_response = call_openai_api(
                prompt=prompt,
                model=model_name,
                max_tokens=2000,
                temperature=0.2
            )
            codeframe_text = api_response.choices[0].message.content.strip()
            codeframe_text = codeframe_text.replace("```json", "").replace("```", "")
            batch_codeframe = validate_json_response(codeframe_text)
            if batch_codeframe is None:
                return {}
            
            # Convert string keywords to lists if needed
            for code_name, details in batch_codeframe.items():
                if isinstance(details.get("keywords", []), str):
                    details["keywords"] = [k.strip() for k in details["keywords"].split(",")]
            
            return batch_codeframe
        except Exception as e:
            st.error(f"Error generating codeframe: {e}")
            return {}

def process_all_responses_for_question(responses, question_text, num_codes=10, batch_size=100):
    responses = list(set(responses))
    total_responses = len(responses)
    if total_responses <= batch_size:
        codeframe = generate_codeframe_batch(responses, question_text, num_codes)
        return update_codeframe({}, codeframe)
    else:
        np.random.shuffle(responses)
        global_codeframe = {}
        total_batches = len(responses) // batch_size + (1 if len(responses) % batch_size != 0 else 0)
        progress_bar = st.progress(0)
        status_container = st.empty()
        for batch_index in range(total_batches):
            status_container.markdown(f"**Processing Batch {batch_index+1}/{total_batches}**")
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size
            batch = responses[start_idx:end_idx]
            batch_codeframe = generate_codeframe_batch(batch, question_text, num_codes)
            global_codeframe = update_codeframe(global_codeframe, batch_codeframe)
            progress_bar.progress((batch_index+1) / total_batches)
            time.sleep(1)
        progress_bar.empty()
        status_container.success("✅ All batches processed successfully!")
        return global_codeframe

def assign_codes_for_question(responses, question_text, codeframe):
    results = []
    code_definitions = "\n".join([f"{code}: {details['description']}" for code, details in codeframe.items()])
    local_api_key = st.session_state.get("api_key", "")
    
    def process_response(response_text):
        prompt = f"""[Question Text]:
{question_text}

[Codeframe]:
{code_definitions}

[Verbatim]:
"{response_text}"

Using the above information, assign the most appropriate code number(s) from the codeframe to this response.
If multiple codes apply, list them separated by commas.
Output only a valid JSON object with exactly the following keys:
- "codes": a list of code numbers assigned,
- "confidence": a list of confidence scores (0-100) for each code,
- "reasoning": a brief explanation for the code assignment.
IMPORTANT: Output ONLY the valid JSON object with no extra text.
Assistant:"""
        try:
            model_name = st.session_state.get("current_model", "gpt-4o-mini")
            api_response = call_openai_api(
                prompt=prompt,
                model=model_name,
                max_tokens=5000,
                temperature=0.1,
                api_key=local_api_key
            )
            response_str = api_response.choices[0].message.content.strip()
            assignment = json.loads(response_str)
            return {
                "response": response_text,
                "codes": assignment.get("codes", []),
                "confidence": assignment.get("confidence", []),
                "reasoning": assignment.get("reasoning", "")
            }
        except Exception as e:
            return {
                "response": response_text,
                "codes": [999],  # Changed from "Error" to numeric code
                "confidence": [0],
                "reasoning": f"Error: {str(e)}"
            }
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_response, r) for r in responses]
        for future in as_completed(futures):
            results.append(future.result())
    return pd.DataFrame(results)

def generate_wordcloud(responses):
    text = " ".join(responses)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def generate_topic_modeling_for_question(responses, num_topics=5):
    valid_texts = [str(t) for t in responses if str(t).strip()]
    if len(valid_texts) < 10:
        st.warning("Not enough responses for topic modeling")
        return None, None
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(valid_texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
    lda_model.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topic_list = []
    for idx, topic in enumerate(lda_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topic_list.append({
            "Topic": f"Topic {idx+1}",
            "Keywords": ", ".join(topic_words),
            "Weight": float(topic.sum() / lda_model.components_.sum())
        })
    topic_df = pd.DataFrame(topic_list)
    return topic_df, lda_model

# ---------------------------
# Sidebar - Enhanced Layout
# ---------------------------
with st.sidebar:
    st.markdown("## 🔑 Setup & Data")
    with st.container():
        st.markdown("### API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", 
                              help="Get your API key from platform.openai.com",
                              placeholder="sk-...")
        
        if api_key:
            st.session_state.api_key = api_key
            if init_openai_client(api_key):
                st.success("✅ API Connected", icon="🔒")

    with st.container():
        st.markdown("### Data Upload")
        uploaded_file = st.file_uploader("Upload Survey Data", 
                                       type=["xlsx"],
                                       help="Required sheets: 'Survey_OE_Q' and 'verbatims'")
        if uploaded_file:
            try:
                xls = pd.ExcelFile(uploaded_file)
                if {"Survey_OE_Q", "verbatims"}.issubset(xls.sheet_names):
                    with st.spinner("📂 Loading data..."):
                        df_questions = pd.read_excel(uploaded_file, sheet_name="Survey_OE_Q")
                        df_verbatims = pd.read_excel(uploaded_file, sheet_name="verbatims")
                        st.session_state.questions = df_questions
                        st.session_state.verbatims = df_verbatims
                    st.success("✅ Data loaded successfully!")
                else:
                    st.error("⚠️ Missing required sheets: 'Survey_OE_Q' and 'verbatims'")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

# ---------------------------
# Main Interface - Modern Layout
# ---------------------------
st.title("📈 Survey Auto-Coding Tool")
st.markdown("---")

# Initialize question dictionary
question_dict = {}
if st.session_state.questions is not None:
    question_options = st.session_state.questions["question_code"].tolist()
    question_labels = st.session_state.questions["question_label"].tolist()
    question_dict = dict(zip(question_options, question_labels))

# Question Selection Card
with st.container():
    st.markdown("<h2 style='color: black;'>🎯 Select Survey Question</h2>", unsafe_allow_html=True)
    if st.session_state.questions is not None:
        selected_question = st.selectbox(
        <h4 style='color: black;'>Choose a Question</h4>", unsafe_allow_html=True),
            options=question_options,
            format_func=lambda x: f"{x}: {question_dict.get(x, '')}",
            help="Select which open-ended question to analyze"
        )
    else:
        st.info("ℹ️ Upload a file to begin analysis", icon="📤")

# ---------------------------
# Main Analysis Dashboard
# ---------------------------
if (st.session_state.verbatims is not None and 
    not st.session_state.verbatims.empty and 
    selected_question and 
    selected_question in question_dict):
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Auto-Coding", "📦 Results", "📚 Exports"])
    
    with tab1:  # Overview Tab
        with st.container():
            st.markdown("### 📈 Response Statistics")
            col1, col2, col3 = st.columns(3)
            responses = st.session_state.verbatims[selected_question].dropna().astype(str).tolist()
            
            with col1:
                st.metric("Total Responses", len(responses))
            with col2:
                st.metric("Unique Responses", len(set(responses)))
            with col3:
                st.metric("Average Length", f"{np.mean([len(r) for r in responses]):.1f} chars")
            
            st.markdown("### 📋 Response Preview")
            st.dataframe(pd.DataFrame(responses, columns=["Responses"]).head(10), 
                        use_container_width=True,
                        height=300)
            
            st.markdown("### 🌈 Word Cloud")
            if st.button("Generate Word Cloud"):
                with st.spinner("Creating visualization..."):
                    fig = generate_wordcloud(responses)
                    st.pyplot(fig)

    with tab2:  # Coding Tab
        with st.container():
            st.markdown("### 🛠 Coding Tools")
            with st.expander("🧠 Automatic Codeframe Generation", expanded=True):
                st.markdown("**AI-powered codeframe creation**")
                if st.button("🚀 Generate Codeframe", key="gen_codeframe"):
                    if not st.session_state.api_key:
                        st.error("❌ API key required")
                    else:
                        with st.spinner("Analyzing responses..."):
                            default_num_codes = 10
                            if len(responses) <= 100:
                                codeframe_raw = generate_codeframe_batch(
                                    responses, 
                                    question_text=question_dict.get(selected_question, ""), 
                                    num_codes=default_num_codes
                                )
                                codeframe = update_codeframe({}, codeframe_raw)
                            else:
                                codeframe = process_all_responses_for_question(
                                    responses, 
                                    question_text=question_dict.get(selected_question, ""), 
                                    num_codes=default_num_codes, 
                                    batch_size=100
                                )
                            st.session_state.codeframes[selected_question] = codeframe
                        st.success("✅ Codeframe generated!")
                        display_codeframe(codeframe)
            
            with st.expander("🔖 Assign Codes to Responses", expanded=True):
                st.markdown("**Automated coding using generated codeframe**")
                if st.button("📝 Start Coding", key="assign_codes"):
                    if selected_question not in st.session_state.codeframes:
                        st.error("❌ Generate codeframe first")
                    else:
                        with st.spinner("Coding responses..."):
                            codeframe = st.session_state.codeframes[selected_question]
                            df_coded = assign_codes_for_question(
                                responses, 
                                question_text=question_dict.get(selected_question, ""), 
                                codeframe=codeframe
                            )
                            st.session_state.coded_data[selected_question] = df_coded
                        st.success(f"✅ Coded {len(df_coded)} responses!")
                        st.dataframe(df_coded.head(100))
            
            with st.expander("🧩 Topic Modeling", expanded=True):
                st.markdown("**Discover latent themes in responses**")
    # Changed key here
    if st.button("🌌 Run Topic Analysis", key="run_topic_model"):  
        with st.spinner("Analyzing topics..."):
            topic_df, lda_model = generate_topic_modeling_for_question(responses)
            if topic_df is not None:
                st.session_state.topic_model[selected_question] = topic_df
                st.success("✅ Topic modeling complete!")
                st.dataframe(topic_df)
            else:
                st.error("❌ Failed to generate topics")

    with tab3:  # Results Tab
        with st.container():
            st.markdown("### 📋 Coding Results")
            if selected_question in st.session_state.coded_data:
                df_coded = st.session_state.coded_data[selected_question].copy()
                df_coded['codes'] = df_coded['codes'].apply(
                    lambda x: [c for c in x if isinstance(c, (int, float))]
                )
                st.markdown("#### Code Distribution")
                code_counts = pd.Series([item for sublist in df_coded['codes'] for item in sublist]).value_counts()
                fig = px.pie(code_counts, 
                            names=code_counts.index, 
                            values=code_counts.values,
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Coded Responses Preview")
                st.dataframe(df_coded.head(100), use_container_width=True, height=600)

    with tab4:  # Exports Tab
        with st.container():
            st.markdown("### 📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Codeframe Export")
                if selected_question in st.session_state.codeframes:
                    output_cf_filename = f"codeframe_{selected_question}.xlsx"
                    with pd.ExcelWriter(output_cf_filename, engine="xlsxwriter") as writer:
                        codeframe = st.session_state.codeframes[selected_question]
                        codeframe_df = pd.DataFrame([{
                            "Code Number": code,
                            "Code Name": details.get("code_name", ""),
                            "Description": details.get("description", ""),
                            "Keywords": ", ".join(details.get("keywords", []))
                        } for code, details in codeframe.items()])
                        header_info = pd.DataFrame({
                            "Question Code": [selected_question],
                            "Question Label": [question_dict.get(selected_question, "")]
                        })
                        header_info.to_excel(writer, sheet_name="Codeframe", index=False, startrow=0)
                        codeframe_df.to_excel(writer, sheet_name="Codeframe", index=False, startrow=3)
                    with open(output_cf_filename, "rb") as f:
                        st.download_button(
                            "💾 Download Codeframe",
                            data=f,
                            file_name=output_cf_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            with col2:
                st.markdown("#### Full Dataset Export")
                if selected_question in st.session_state.coded_data:
                    def expand_codes(row):
                        codes = row["codes"] if isinstance(row["codes"], list) else []
                        max_len = 5
                        codes = codes + [""] * (max_len - len(codes))
                        return pd.Series(codes, index=[f"Code{i+1}" for i in range(max_len)])
                    
                    df_coded = st.session_state.coded_data[selected_question]
                    codes_expanded = df_coded.apply(expand_codes, axis=1)
                    df_export = pd.concat([df_coded["response"], codes_expanded], axis=1)
                    
                    output_filename = f"coded_output_{selected_question}.xlsx"
                    with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
                        df_export.to_excel(writer, sheet_name="Coded Responses", index=False)
                        codeframe = st.session_state.codeframes.get(selected_question, {})
                        codeframe_df = pd.DataFrame([{
                            "Code Number": code,
                            "Code Name": details.get("code_name", ""),
                            "Description": details.get("description", ""),
                            "Keywords": ", ".join(details.get("keywords", []))
                        } for code, details in codeframe.items()])
                        header_info = pd.DataFrame({
                            "Question Code": [selected_question],
                            "Question Label": [question_dict.get(selected_question, "")]
                        })
                        header_info.to_excel(writer, sheet_name="Codeframe", index=False, startrow=0)
                        codeframe_df.to_excel(writer, sheet_name="Codeframe", index=False, startrow=3)
                    with open(output_filename, "rb") as f:
                        st.download_button(
                            "💾 Download Full Dataset",
                            data=f,
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
else:
    st.empty()

# Footer
st.markdown("---")
current_year = datetime.datetime.now().year
st.markdown(f"<div style='text-align: center; color: #666;'>© {current_year} Survey Open-ended Coding Automation Tool</div>", unsafe_allow_html=True)
