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
                max_tokens=10,
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
    prompt = f"""CODEFRAME CREATION TASK:
                Objective: Generate a detailed codeframe to categorize open-ended survey responses regarding "{question_text}".
                Context:
                - Survey Goal: Identify pain points in user experience.
                - Sample Question: "{question_text}"

                Instructions:
                1. Create a codeframe that organizes responses into high-level themes, subthemes, and individual codes.
                2. For each code, provide:
                - Code Name: A concise label (e.g., "Pricing Concerns").
                - Description: A clear definition explaining the scope (e.g., "Mentions of affordability, value, or cost").
                - Inclusion Criteria: Conditions under which this code should be applied (e.g., "Responses referencing price comparisons").
                - Exclusion Criteria: Conditions under which the code should NOT be applied (e.g., "Mentions of product features without pricing context").
                - Example Response: A real or hypothetical quote illustrating the code (e.g., "It’s too expensive compared to competitors").
                - Sentiment: Indicate if the response tone is Positive, Neutral, or Negative (if applicable).
                3. Special Cases:
                - Include a "Miscellaneous/Other" category for rare or ambiguous responses.
                - Include a "Non-Applicable" code for irrelevant or unintelligible answers.
                4. Process:
                - Step 1: List all possible codes from sample responses. For context, include the first 500 characters of responses below:
                    {responses_text[:500]}
                - Step 2: Group the codes into logical themes and subthemes.
                - Step 3: Refine the codes to avoid redundancy and ensure they are mutually exclusive.
                - Step 4: Validate that the codes cover at least 95% of hypothetical responses.

                Mandatory Codes (apply as verbatim matches):
                97 - Nothing: Empty responses, "nothing", "nada"
                98 - No Answer: "no comment", "no opinion"
                99 - Don't Know: "don't know", "unsure", "dk"

                OUTPUT FORMAT:
                Return a JSON array of objects, where each object represents a row in a table with the following keys:
                - "Theme"
                - "Subtheme"
                - "Code Name"
                - "Description"
                - "Example Response"
                - "Sentiment"

                IMPORTANT:
                - Use double quotes for all keys and string values.
                - Do not include any trailing commas or comments.
                - Ensure the JSON is valid and all brackets are properly closed.
                """
    with st.spinner("Generating codeframe..."):
        try:
            model_name = st.session_state.get("current_model", "gpt-4o-mini")
            api_response = call_openai_api(
                prompt=prompt,
                model=model_name,
                max_tokens=3500,
                temperature=0.2
            )
            codeframe_text = api_response.choices[0].message.content.strip()
            codeframe_text = codeframe_text.replace("```json", "").replace("```", "")
            batch_codeframe = validate_json_response(codeframe_text)
            if batch_codeframe is None:
                return {}
            
            # (Optional) Process the response if any post-conversion adjustments are needed
            return batch_codeframe
        except Exception as e:
            st.error(f"Error generating codeframe: {e}")
            return {}


def process_all_responses_for_question(responses, question_text, num_codes=10, batch_size=200):
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
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                prompt = f"""CODING TASK: Evaluate the following survey response and assign the appropriate codes:"{response_text}"
                            CONTEXT:
                            - This response is provided in the context of the survey question: "{question_text}".
                            - A detailed codeframe with specific code definitions is provided below. Each code includes precise keywords/phrases, scope definitions, and inclusion/exclusion criteria.
                            - Mandatory Codes:
                                * 97 - Nothing: For empty or meaningless responses (e.g., "nothing", "nada").
                                * 98 - No Answer: For responses indicating refusal (e.g., "no comment", "no opinion").
                                * 99 - Don't Know: For responses indicating uncertainty (e.g., "don't know", "unsure", "dk").

                            INSTRUCTIONS:

                            1. Preliminary Check – Mandatory Codes:
                            - If the response is empty or contains words like "nothing" or "nada", immediately assign code 97.
                            - If the response contains a refusal (e.g., "no comment" or "no opinion"), assign code 98.
                            - If the response expresses uncertainty (e.g., "don't know", "unsure", "dk"), assign code 99.
                            - In such cases, provide a brief explanation and do not evaluate further.

                            2. Substantive Coding:
                            - Review the provided code definitions carefully:
                            {code_definitions}
                            - For each code, compare the response text against its keywords/phrases and scope:
                                * If an exact keyword or phrase is present, consider that a 100% confidence match.
                                * If a partial keyword or a variant is present in a relevant context, consider that an 80% confidence match.
                                * If the meaning is only implied by the response, assign a 50% confidence match.
                            - Ensure that the selected codes are mutually exclusive and collectively cover the response.
                            - If no substantive code is applicable, assign code 999 with 0% confidence to indicate a non-match.

                            3. Confidence Evaluation:
                            - For each assigned code, determine a numerical confidence level (0 to 100) based on the strength of the match:
                                * 100% for exact phrase matches.
                                * 80% for partial matches with strong contextual evidence.
                                * 50% for implied meaning.
                            - Each response must have at least one code assigned. If no match is found after thorough analysis, use code 999 with 0% confidence.

                            4. Validation and Reasoning:
                            - Validate that at least one code is assigned.
                            - Provide a concise reasoning (1-2 sentences) that details how each code was selected based on the presence of keywords, context, and any partial or exact matches.
                            - Your reasoning should explain the matching logic clearly, referencing key phrases or context that led to the assignment of each code.

                            OUTPUT FORMAT – YOU MUST FOLLOW EXACTLY:
                            Return a JSON object that exactly adheres to this structure (and nothing else):
                            {{
                                "codes": [list of integers],           // Only numbers between 97 and 999
                                "confidence": [list of integers],        // Confidence levels (0-100) for each code
                                "reasoning": "A brief explanation of the matching logic used"
                            }}

                            GENERAL RULES:
                            - Use double quotes for all keys and string values.
                            - Do not include any additional keys or metadata.
                            - Ensure the output is valid JSON with all brackets properly closed.
                            - Avoid any extra commentary or non-JSON text.
                            - Follow the instructions precisely and return only the JSON object as specified.

                            EXAMPLE:
                            If the response is "I really love the fast service but found the pricing a bit steep", a possible output might be:
                            {{
                                "codes": [101, 102],
                                "confidence": [100, 80],
                                "reasoning": "Exact match for 'fast service' aligned with the Service Speed code (101) and a partial match for pricing concerns aligning with code 102."
                            }}

                            Please provide your analysis strictly according to these instructions.
                            """
                model_name = st.session_state.get("current_model", "gpt-4o-mini")
                api_response = call_openai_api(
                    prompt=prompt,
                    model=model_name,
                    max_tokens=2500,
                    temperature=0.1,
                    api_key=local_api_key
                )
                response_str = api_response.choices[0].message.content.strip()

                # Clean and parse response
                response_str = re.sub(r'^[^{]*', '', response_str)  # Remove non-JSON prefixes
                response_str = re.sub(r'[^}]*$', '', response_str)  # Remove non-JSON suffixes
                if response_str.startswith('```json'):
                    response_str = response_str[6:-3].strip()
                
                assignment = json.loads(response_str)

                # Validate response structure
                if not all(key in assignment for key in ["codes", "confidence", "reasoning"]):
                    raise ValueError("Missing required keys in response")

                # Convert codes to integers
                assignment["codes"] = [int(c) for c in assignment["codes"]]

                return {
                    "response": response_text,
                    "codes": assignment["codes"],
                    "confidence": assignment["confidence"],
                    "reasoning": assignment["reasoning"]
                }

            except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
                if attempt < max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                
                # Final fallback after retries
                try:
                    codes = list(map(int, re.findall(r'\b\d{2,3}\b', response_str)))
                    if codes:
                        return {
                            "response": response_text,
                            "codes": codes[:3],
                            "confidence": [min(100, len(str(c))*30) for c in codes[:3]],
                            "reasoning": f"Recovered from error: {str(e)}"
                        }
                except:
                    pass

                return {
                    "response": response_text,
                    "codes": [999],
                    "confidence": [0],
                    "reasoning": f"Critical error: {str(e)}"
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

def generate_topic_names(keywords_list):
    """Generate human-readable topic names using OpenAI"""
    prompt = f"""Analyze these keyword groups and generate a short, descriptive topic name (2-4 words) for each.
    Follow these rules:
    1. Use title case
    2. Be specific and conceptual
    3. No quotation marks
    4. Respond ONLY with comma-separated names
    
    Keywords Groups:
    {chr(10).join([f'Group {i+1}: {", ".join(kw)}' for i, kw in enumerate(keywords_list)])}
    
    Topic Names:"""
    
    try:
        response = call_openai_api(
            prompt=prompt,
            model="gpt-4o-mini",  # Using faster model for this task
            max_tokens=100,
            temperature=0.3
        )
        names = response.choices[0].message.content.strip().split(', ')
        return names
    except Exception as e:
        st.error(f"Topic naming failed: {str(e)}")
        return [f"Topic {i+1}" for i in range(len(keywords_list))]

def generate_topic_modeling_for_question(responses, num_topics=5):
    valid_texts = [str(t) for t in responses if str(t).strip()]
    if len(valid_texts) < 10:
        st.warning("Not enough responses for topic modeling")
        return None, None

    # Original LDA implementation
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(valid_texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
    lda_model.fit(dtm)
    
    # Extract keywords
    feature_names = vectorizer.get_feature_names_out()
    keywords_list = []
    for topic in lda_model.components_:
        keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        keywords_list.append(keywords)

    # Generate AI-powered topic names
    with st.spinner("🧠 Generating meaningful topic names..."):
        topic_names = generate_topic_names(keywords_list)

    # Build enhanced dataframe
    topic_list = []
    for idx, (name, keywords, topic_weights) in enumerate(zip(topic_names, keywords_list, lda_model.components_)):
        topic_list.append({
            "Topic Number": idx+1,
            "Topic Name": name,
            "Keywords": ", ".join(keywords[:10]),  # Show top 10 keywords
            "Keyword Weights": [float(weight) for weight in topic_weights[:10]],  # Raw weights for analysis
            "Topic Weight": float(topic_weights.sum() / lda_model.components_.sum() * 100)
        })

    return pd.DataFrame(topic_list), lda_model

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
    if st.session_state.questions is not None:
        selected_question = st.selectbox(
        "**Choose a Question**",
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
            col1, col2= st.columns(2)
            responses = st.session_state.verbatims[selected_question].dropna().astype(str).tolist()
            
            with col1:
                st.metric("Total Responses", len(responses))
            with col2:
                st.metric("Unique Responses", len(set(responses)))
            
            st.markdown("### 📋 Response Preview")
            st.dataframe(pd.DataFrame(responses, columns=["Responses"]).head(10), 
                         use_container_width=True,
                         height=300)
            
            st.markdown("### 🌈 Word Cloud")
            if st.button("Generate Word Cloud"):
                with st.spinner("Creating visualization..."):
                    fig = generate_wordcloud(responses)
                    st.pyplot(fig)

    with tab2:  # Auto-Coding Tab
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
                            if len(responses) <= 200:
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
                                    batch_size=200
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
                        st.dataframe(df_coded.head(30))
            
            with st.expander("🧩 Topic Modeling", expanded=True):
                st.markdown("**Discover latent themes in responses**")
                if st.button("🌌 Run Topic Analysis", key="run_topic_model"):
                    with st.spinner("Analyzing topics..."):
                        topic_df, lda_model = generate_topic_modeling_for_question(responses)
                    
                    if topic_df is not None:
                        st.session_state.topic_model[selected_question] = topic_df
                        st.success("✅ Topic modeling complete!")
                        
                        # Display enhanced dataframe
                        display_df = topic_df[["Topic Number", "Topic Name", "Keywords", "Topic Weight"]]
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=200,
                            column_config={
                                "Topic Name": "AI-Generated Theme",
                                "Keywords": "Top Keywords", 
                                "Topic Weight": st.column_config.NumberColumn(
                                    "Prevalence (%)",
                                    format="%.2f",
                                    help="Percentage of responses containing this theme"
                                )
                            }
                        )

                        # Update the visualization code:
                        fig = px.bar(
                            topic_df,
                            x='Topic Name',
                            y='Topic Weight',
                            labels={'Topic Weight': 'Prevalence (%)'},
                            title="Topic Prevalence Distribution"
                        )

    with tab3:  # Results Tab
        with st.container():
            st.markdown("### 📋 Coding Results")
            if selected_question in st.session_state.coded_data:
                df_coded = st.session_state.coded_data[selected_question].copy()
                df_coded['codes'] = df_coded['codes'].apply(
                    lambda x: [c for c in x if isinstance(c, (int, float))]
                )
                st.markdown("#### Code Distribution")
                
                # Get code counts and codeframe
                code_counts = pd.Series(
                    [item for sublist in df_coded['codes'] for item in sublist]
                ).value_counts()
                codeframe = st.session_state.codeframes.get(selected_question, {})
                
                # Map code numbers to code names
                code_names = []
                for code_num in code_counts.index:
                    if code_num == 999:
                        code_names.append("Processing Error")
                    else:
                        # Use get() with default value
                        code_info = codeframe.get(int(code_num), {})
                        code_names.append(
                            code_info.get("code_name", f"Code {code_num}")
                        )

                # Create DataFrame for visualization
                df_pie = pd.DataFrame({
                    "Code Name": code_names,
                    "Count": code_counts.values
                })

                threshold = 1  # Percentage threshold
                df_pie['Code Name'] = np.where(
                    df_pie['Count']/df_pie['Count'].sum() * 100 < threshold,
                    'Other',
                    df_pie['Code Name']
                )
                df_pie = df_pie.groupby('Code Name', as_index=False).sum()
                
                # Generate enhanced pie chart
                fig = px.pie(
                    df_pie,
                    names='Code Name',
                    values='Count',
                    hole=0.4,
                    title="Code Distribution",
                    labels={'Count': 'Responses'},
                    width=1000,  # Increased width
                    height=800   # Increased height
                )

                # Update layout for better label visibility
                fig.update_layout(
                    uniformtext_minsize=12,  # Minimum font size
                    uniformtext_mode='hide',  # Hide labels that don't fit
                    margin=dict(t=50, b=50, l=20, r=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                # Force show all labels with leader lines
                fig.update_traces(
                    textposition='outside',
                    textinfo='percent+label',
                    insidetextorientation='auto',
                    pull=[0.02] * len(df_pie),  # Small pull for separation
                    marker=dict(line=dict(color='#ffffff', width=2)))
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Coded Responses Preview")
                st.dataframe(df_coded.head(20), use_container_width=True, height=600)

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
                        
                        # Add error code if present in data
                        if selected_question in st.session_state.coded_data:
                            df_coded = st.session_state.coded_data[selected_question]
                            if 999 in df_coded['codes'].explode().unique():
                                error_code = {
                                    "Code Number": 999,
                                    "Code Name": "Processing Error",
                                    "Description": "Failed to code this response",
                                    "Keywords": ""
                                }
                                codeframe_df = pd.concat([codeframe_df, pd.DataFrame([error_code])])
                        
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
                        try:
                            # Get original verbatims and coded data
                            original_verbatims = st.session_state.verbatims.copy()
                            coded_data = st.session_state.coded_data[selected_question].copy()
                            
                            # Ensure we maintain original structure
                            final_export = original_verbatims.copy()
                            
                            # Add coding results as new columns
                            final_export['Coding_Codes'] = coded_data['codes'].apply(
                                lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
                            )
                            final_export['Coding_Confidence'] = coded_data['confidence']
                            final_export['Coding_Reasoning'] = coded_data['reasoning']
                            
                            # Add individual code columns (Code1-Code5)
                            def expand_codes(row):
                                codes = coded_data.at[row.name, 'codes'] if row.name in coded_data.index else []
                                if pd.isna(codes):
                                    codes = []
                                elif not isinstance(codes, list):
                                    codes = [codes]
                                
                                # Clean and format codes
                                clean_codes = []
                                for code in codes:
                                    if pd.notna(code):
                                        if isinstance(code, float) and code.is_integer():
                                            clean_codes.append(str(int(code)))
                                        else:
                                            clean_codes.append(str(code))
                                
                                # Pad to 5 columns
                                return pd.Series(
                                    clean_codes[:5] + [''] * (5 - len(clean_codes[:5])),
                                    index=[f'Code{i+1}' for i in range(5)]
                                )
                            
                            # Add code columns directly to final export
                            code_columns = final_export.apply(expand_codes, axis=1)
                            final_export = pd.concat([final_export, code_columns], axis=1)
                            
                            # Verify all original columns are present
                            original_columns = set(st.session_state.verbatims.columns)
                            current_columns = set(final_export.columns)
                            missing_columns = original_columns - current_columns
                            if missing_columns:
                                raise ValueError(f"Missing columns in export: {', '.join(missing_columns)}")
                            
                            # Generate filename
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"Coded_{selected_question}_{timestamp}.xlsx"
                            
                            # Create Excel file
                            with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
                                # Write main data
                                final_export.to_excel(
                                    writer,
                                    sheet_name='Coded Responses',
                                    index=False
                                )
                                
                                # Create codeframe documentation
                                codeframe_sheet = writer.book.add_worksheet('Codeframe')
                                
                                # Write header information
                                codeframe_sheet.write(0, 0, 'Question Code')
                                codeframe_sheet.write(0, 1, selected_question)
                                codeframe_sheet.write(1, 0, 'Question Label')
                                codeframe_sheet.write(1, 1, question_dict.get(selected_question, ''))
                                
                                # Write codeframe data
                                codeframe_data = []
                                for code_num, details in st.session_state.codeframes.get(selected_question, {}).items():
                                    codeframe_data.append({
                                        'Code Number': code_num,
                                        'Code Name': details.get('code_name', ''),
                                        'Description': details.get('description', ''),
                                        'Keywords': ', '.join(details.get('keywords', []))
                                    })
                                
                                # Add error code if present
                                if 999 in coded_data['codes'].explode().unique():
                                    codeframe_data.append({
                                        'Code Number': 999,
                                        'Code Name': 'Processing Error',
                                        'Description': 'Failed to code this response',
                                        'Keywords': ''
                                    })
                                
                                # Convert to DataFrame and write
                                if codeframe_data:
                                    pd.DataFrame(codeframe_data).to_excel(
                                        writer,
                                        sheet_name='Codeframe',
                                        startrow=3,
                                        index=False
                                    )
                            
                            # Create download button
                            with open(output_filename, 'rb') as f:
                                st.download_button(
                                    label='📥 Download Full Dataset',
                                    data=f,
                                    file_name=output_filename,
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    help='Includes all original columns plus coding results'
                                )
                            
                            # Clean up temporary file
                            os.remove(output_filename)
                            
                        except Exception as e:
                            st.error(f'Export failed: {str(e)}')
                            st.error('Please ensure the data structure matches expectations')
                            
                            # Debug information
                            with st.expander('Show Debug Details'):
                                st.write('### Original Verbatims Columns')
                                st.write(list(st.session_state.verbatims.columns))
                                
                                st.write('### Processed Export Columns')
                                if 'final_export' in locals():
                                    st.write(list(final_export.columns))
                                else:
                                    st.write('Export not initialized')

# Footer
st.markdown("---")
current_year = datetime.datetime.now().year
st.markdown(f"<div style='text-align: center; color: #666;'>© {current_year} Survey Open-ended Coding Automation Tool</div>", unsafe_allow_html=True)

