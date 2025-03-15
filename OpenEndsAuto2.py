import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import numpy as np
import os, json, time, re, ast, sqlite3, bcrypt
from openai import OpenAI
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
    'authenticated': False,
    'username': None,
    'subscription_active': False,
    'subscription_expiry': None,
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
# Database Initialization
# ---------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            subscription_plan TEXT,
            subscription_expiry DATE,
            active BOOLEAN DEFAULT TRUE
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# ---------------------------
# Authentication Functions
# ---------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_user(username, password, subscription_plan):
    hashed_pw = hash_password(password)
    expiry_date = datetime.datetime.now() + datetime.timedelta(days=30)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users 
            (username, password_hash, subscription_plan, subscription_expiry, active)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hashed_pw, subscription_plan, expiry_date.date(), True))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username exists
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user:
        return {
            'username': user[0],
            'password_hash': user[1],
            'subscription_plan': user[2],
            'subscription_expiry': user[3],
            'active': user[4]
        }
    return None

def is_subscription_active(expiry_date):
    if expiry_date:
        if isinstance(expiry_date, str):
            expiry_date = datetime.datetime.strptime(expiry_date, '%Y-%m-%d').date()
        return expiry_date >= datetime.date.today()
    return False

# ---------------------------
# Authentication Flow
# ---------------------------
if not st.session_state.authenticated:
    st.title("Welcome to Survey Coding Assistant")
    auth_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    # Login Tab
    with auth_tab:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                user = get_user(username)
                if user and check_password(user['password_hash'], password):
                    if is_subscription_active(user['subscription_expiry']):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.subscription_active = True
                        st.session_state.subscription_expiry = user['subscription_expiry']
                        st.rerun()
                    else:
                        st.session_state.temp_username = username
                        st.session_state.show_subscription_renewal = True
                        st.error("Your subscription has expired. Please renew.")
                else:
                    st.error("Incorrect username or password")

    # Signup Tab
    with signup_tab:
        with st.form("Sign Up"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            subscription_plan = st.selectbox("Subscription Plan", 
                ["Basic ($9.99/month)", "Premium ($29.99/month)"])
            signup_button = st.form_submit_button("Sign Up")

            if signup_button:
                if not new_username or not new_password:
                    st.error("Please fill in all fields")
                else:
                    existing_user = get_user(new_username)
                    if existing_user:
                        st.error("Username already exists. Please choose another.")
                    else:
                        success = create_user(new_username, new_password, subscription_plan)
                        if success:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Error creating account. Please try again.")

    # Subscription Renewal
    if st.session_state.get('show_subscription_renewal'):
        st.subheader("Renew Your Subscription")
        with st.form("Renew Subscription"):
            renew_plan = st.selectbox("Choose a Plan", 
                ["Basic ($9.99/month)", "Premium ($29.99/month)"])
            renew_button = st.form_submit_button("Purchase Renewal")

            if renew_button:
                new_expiry = datetime.date.today() + datetime.timedelta(days=30)
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('''
                    UPDATE users 
                    SET subscription_plan = ?, subscription_expiry = ?, active = ?
                    WHERE username = ?
                ''', (renew_plan, new_expiry, True, st.session_state.temp_username))
                conn.commit()
                conn.close()
                
                st.session_state.authenticated = True
                st.session_state.username = st.session_state.temp_username
                st.session_state.subscription_active = True
                st.session_state.subscription_expiry = new_expiry
                st.success("Subscription renewed! Redirecting...")
                time.sleep(1)
                st.rerun()

    st.stop()  # Stop execution if not authenticated

# ---------------------------
# Main Application (Only accessible when authenticated)
# ---------------------------

# Add authentication info to sidebar
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.username}")
    st.markdown(f"**Subscription:** {st.session_state.subscription_expiry}")
    
    if st.button("🗝 Logout"):
        for key in session_defaults:
            if key != 'codeframes' and key != 'coded_data':  # Preserve work data
                st.session_state[key] = session_defaults[key]
        st.rerun()
    
    if st.button("📅 Manage Subscription"):
        st.session_state.show_subscription_management = True

# Subscription Management
if st.session_state.get('show_subscription_management'):
    with st.container():
        st.subheader("Subscription Management")
        current_expiry = st.session_state.subscription_expiry
        if is_subscription_active(current_expiry):
            st.success(f"✅ Active until {current_expiry}")
        else:
            st.error("❌ Subscription expired")
        
        with st.form("Upgrade Subscription"):
            new_plan = st.selectbox("Choose New Plan", 
                ["Basic ($9.99/month)", "Premium ($29.99/month)"])
            upgrade_button = st.form_submit_button("Upgrade Plan")
            
            if upgrade_button:
                new_expiry = datetime.date.today() + datetime.timedelta(days=30)
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('''
                    UPDATE users 
                    SET subscription_plan = ?, subscription_expiry = ?
                    WHERE username = ?
                ''', (new_plan, new_expiry, st.session_state.username))
                conn.commit()
                conn.close()
                
                st.session_state.subscription_expiry = new_expiry
                st.success("Subscription updated successfully!")
                time.sleep(1)
                st.session_state.show_subscription_management = False
                st.rerun()

# ---------------------------
# Helper Functions (Reverted to OpenAI)
# ---------------------------
def call_openai_api(prompt, model, max_tokens, temperature, stop_sequences=None, api_key=None):
    client = st.session_state.get("client")
    if not client:
        raise Exception("OpenAI client not initialized. Check your API key.")
    try:
        response = client.chat.completions.create(
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
        client = OpenAI(api_key=api_key)
        test_prompt = "Hello OpenAI!"
        model_name = "gpt-4o-mini"
        with st.spinner("Testing API connection..."):
            _ = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=1,
                temperature=0.0
            )
        st.session_state.client = client
        st.session_state.current_model = model_name
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return False

def validate_json_response(response_text):
    try:
        # Remove all JSON markdown formatting
        cleaned = re.sub(r'^```(json)?\s*|\s*```$', '', response_text, flags=re.DOTALL)
        
        # Fix common JSON issues
        cleaned = cleaned.strip()
        cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', cleaned)  # Remove invalid escapes
        cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
        cleaned = re.sub(r',\s*]', ']', cleaned)   # Remove trailing commas
        cleaned = re.sub(r'(?<={|,)\s*([a-zA-Z_]+)\s*:', r'"\1":', cleaned)  # Add quotes to keys
        
        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to find JSON object in response
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            
            # Try literal eval as fallback
            return ast.literal_eval(cleaned)
            
    except Exception as e:
        st.error(f"JSON validation failed: {str(e)}\nCleaned response: {cleaned}\nRaw response: {response_text}")
        return None

def update_codeframe(global_codeframe, batch_codeframe):
    """Process batch codeframe into the global codeframe storage"""
    # Convert both list and dict responses to standardized format
    processed_codes = []
    
    if isinstance(batch_codeframe, list):
        # Directly use list format from API response
        processed_codes = batch_codeframe
    elif isinstance(batch_codeframe, dict):
        # Convert old dict format to list format
        processed_codes = [
            {
                "Theme": details.get("Theme", ""),
                "Subtheme": details.get("Subtheme", ""),
                "Code Name": code_name,
                "Description": details.get("Description", ""),
                "Example Response": details.get("Example Response", ""),
                "Sentiment": details.get("Sentiment", "Neutral")
            }
            for code_name, details in batch_codeframe.items()
        ]
    
    # Add codes to global codeframe with unique numeric IDs
    for code_entry in processed_codes:
        # Check for existing code by name
        exists = any(
            entry["Code Name"] == code_entry["Code Name"]
            for entry in global_codeframe.values()
        )
        
        if not exists and code_entry["Code Name"]:
            code_number = st.session_state.code_counter
            st.session_state.code_counter += 1
            
            # Map to standardized format
            global_codeframe[code_number] = {
                "Theme": code_entry.get("Theme", "Uncategorized"),
                "Subtheme": code_entry.get("Subtheme", ""),
                "Code Name": code_entry["Code Name"],
                "Description": code_entry.get("Description", ""),
                "Example Response": code_entry.get("Example Response", ""),
                "Sentiment": code_entry.get("Sentiment", "Neutral"),
                "Keywords": code_entry.get("Keywords", [])
            }
    
    return global_codeframe

def display_codeframe(codeframe):
    """Display codeframe in the desired dataframe format"""
    if not codeframe:
        st.warning("No codeframe generated yet")
        return
    
    # Create dataframe from the standardized format
    df = pd.DataFrame([
        {
            "Theme": details["Theme"],
            "Subtheme": details["Subtheme"],
            "Code Name": details["Code Name"],
            "Description": details["Description"],
            "Example Response": details["Example Response"],
            "Sentiment": details["Sentiment"]
        }
        for code_num, details in codeframe.items()
    ])
    
    # Add error code row
    error_row = {
        "Theme": "System",
        "Subtheme": "Errors",
        "Code Name": 999,
        "Description": "Failed to code this response",
        "Example Response": "",
        "Sentiment": "Neutral"
    }
    df = pd.concat([df, pd.DataFrame([error_row])], ignore_index=True)
    
    st.dataframe(
        df,
        use_container_width=True,
        column_order=["Theme", "Subtheme", "Code Name", "Description", "Example Response", "Sentiment"],
        hide_index=True
    )

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
                max_tokens=1000,
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
        
# Add this function in the Helper Functions section
def process_all_responses_for_question(responses, question_text, num_codes=10, batch_size=200):
    """Process large response sets in batches and aggregate results"""
    global_codeframe = {}
    
    # Split responses into batches
    batches = [responses[i:i+batch_size] 
               for i in range(0, len(responses), batch_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num, batch_responses in enumerate(batches):
        try:
            status_text.text(f"Processing batch {batch_num+1}/{len(batches)}...")
            batch_codeframe = generate_codeframe_batch(
                batch_responses, 
                question_text, 
                num_codes
            )
            global_codeframe = update_codeframe(global_codeframe, batch_codeframe)
            progress_bar.progress((batch_num+1)/len(batches))
        except Exception as e:
            st.error(f"Error processing batch {batch_num+1}: {str(e)}")
            continue
    
    status_text.empty()
    progress_bar.empty()
    return global_codeframe


def assign_codes_for_question(responses, question_text, codeframe):
    """Assign codes to responses using OpenAI API with parallel processing"""
    results = []
    code_definitions = "\n".join([f"{code}: {details['Description']}" for code, details in codeframe.items()])
    client = st.session_state.get("client")
    
    if client is None:
        st.error("OpenAI client is not initialized. Please check your API key.")
        return pd.DataFrame()

    def process_response(response_text):
        """Process individual response with retry logic"""
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
                            - For each assigned code, determine a numerical confidence level (0 to 100) based on the strength of the match.

                            4. Validation:
                            - Validate that at least one code is assigned.

                            OUTPUT FORMAT – JSON object:
                            {{
                                "codes": [list of integers],   
                                "confidence": [list of integers]
                            }}
                            """

                model_name = st.session_state.get("current_model", "gpt-4o-mini")
                api_response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                response_str = api_response.choices[0].message.content.strip()
                
                # Clean JSON response
                response_str = re.sub(r'^[^{]*', '', response_str)
                response_str = re.sub(r'[^}]*$', '', response_str)
                if response_str.startswith('```json'):
                    response_str = response_str[6:-3].strip()
                
                assignment = json.loads(response_str)

                # Validate response structure
                if not all(key in assignment for key in ["codes", "confidence"]):
                    raise ValueError("Invalid response format")

                # Convert codes to integers
                assignment["codes"] = [int(c) for c in assignment["codes"]]

                return {
                    "response": response_text,
                    "codes": assignment["codes"],
                    "confidence": assignment["confidence"]
                }

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                return {
                    "response": response_text,
                    "codes": [999],
                    "confidence": [0]
                }

    # Parallel processing with progress tracking
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_response, r) for r in responses]
        
        progress_bar = st.progress(0)
        results = []
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                progress_bar.progress((i+1)/len(futures))
            except Exception as e:
                results.append({
                    "response": responses[i],
                    "codes": [999],
                    "confidence": [0]
                })
        progress_bar.empty()

    # Create DataFrame with proper error handling
    try:
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error creating results dataframe: {str(e)}")
        return pd.DataFrame(columns=["response", "codes", "confidence"])


def generate_wordcloud(responses):
    text = " ".join(responses)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def generate_topic_names(keywords_list):
    """Generate human-readable topic names using OpenRouter"""
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
            model="gpt-4o-mini",
            max_tokens=50,
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
            
            # =============================================
            # Codeframe Generation Section
            # =============================================
            # Check if codeframe exists for current question
            codeframe_exists = selected_question in st.session_state.codeframes
            
            # Create expander that auto-expands when new content exists
            with st.expander("🧠 Automatic Codeframe Generation", expanded=codeframe_exists):
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
                            # Store codeframe and force display
                            st.session_state.codeframes[selected_question] = codeframe
                            st.session_state[f"show_codeframe_{selected_question}"] = True
                            st.success("✅ Codeframe generated!")
                            st.rerun()  # Force immediate update
                
                # Display immediately after generation
                if codeframe_exists:
                    display_codeframe(st.session_state.codeframes[selected_question])

            # =============================================
            # Coding Execution Section
            # =============================================
            coded_data_exists = selected_question in st.session_state.coded_data
            
            with st.expander("🔖 Assign Codes to Responses", expanded=coded_data_exists):
                if st.button("📝 Start Coding", key="assign_codes"):
                    if not codeframe_exists:
                        st.error("❌ Generate codeframe first")
                    else:
                        with st.spinner("Coding responses..."):
                            codeframe = st.session_state.codeframes[selected_question]
                            df_coded = assign_codes_for_question(
                                responses, 
                                question_text=question_dict.get(selected_question, ""), 
                                codeframe=codeframe
                            )
                            # Store results and force display
                            st.session_state.coded_data[selected_question] = df_coded
                            st.session_state[f"show_coding_{selected_question}"] = True
                            st.success(f"✅ Coded {len(df_coded)} responses!")
                            st.rerun()  # Force immediate update
                
                # Display immediately after coding
                if coded_data_exists:
                    df_coded = st.session_state.coded_data[selected_question]
                    st.dataframe(
                        df_coded.head(30),
                        column_config={
                            "codes": "Assigned Codes",
                            "confidence": "Confidence"
                        },
                        use_container_width=True,
                    )

            # =============================================
            # Topic Modeling Section
            # =============================================
            with st.expander("🧩 Topic Modeling"):
                if st.button("🌌 Run Topic Analysis", key="run_topic_model"):
                    with st.spinner("Analyzing topics..."):
                        topic_df, lda_model = generate_topic_modeling_for_question(responses)
                    
                    if topic_df is not None:
                        st.session_state.topic_model[selected_question] = topic_df
                        st.success("✅ Topic modeling complete!")
                        
                        display_df = topic_df[["Topic Number", "Topic Name", "Keywords", "Topic Weight"]]
                        st.dataframe(
                            display_df,
                            use_container_width=True,
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

                        fig = px.bar(
                            topic_df,
                            x='Topic Name',
                            y='Topic Weight',
                            labels={'Topic Weight': 'Prevalence (%)'},
                            title="Topic Prevalence Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:  # Results Tab
        with st.container():
            st.markdown("### 📋 Coding Results")
            if selected_question in st.session_state.coded_data:
                df_coded = st.session_state.coded_data[selected_question].copy()
                codeframe = st.session_state.codeframes.get(selected_question, {})
                
                # Extract code labels from codeframe
                def get_code_label(code_num):
                    if code_num == 999:
                        return "Processing Error"
                    code_info = codeframe.get(int(code_num), {})
                    return code_info.get("Code Name", f"Code {code_num}")

                # Process codes and get labels
                df_coded['code_labels'] = df_coded['codes'].apply(
                    lambda codes: [get_code_label(c) for c in codes]
                )

                # Flatten the list of code labels and count occurrences
                all_labels = [label for sublist in df_coded['code_labels'] for label in sublist]
                label_counts = pd.Series(all_labels).value_counts().reset_index()
                label_counts.columns = ['Code Label', 'Count']

                # Create pie chart data with labels
                df_pie = label_counts.copy()
                
                # Threshold for grouping small slices
                threshold = 1  # Percentage threshold
                total = df_pie['Count'].sum()
                df_pie['Percentage'] = df_pie['Count'] / total * 100
                df_pie['Code Label'] = np.where(
                    df_pie['Percentage'] < threshold,
                    'Other',
                    df_pie['Code Label']
                )
                
                # Group small categories
                df_pie = df_pie.groupby('Code Label', as_index=False).agg({
                    'Count': 'sum',
                    'Percentage': 'sum'
                })

                # Create the visualization
                fig = px.pie(
                    df_pie,
                    names='Code Label',
                    values='Count',
                    hole=0.3,
                    title="Code Distribution by Label",
                    labels={'Count': 'Responses'},
                    custom_data=['Percentage']
                )

                # Improve label formatting
                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%",
                    texttemplate='%{label}<br>(%{percent:.1%})',
                    textposition='outside',
                    insidetextorientation='auto'
                )

                fig.update_layout(
                    uniformtext_minsize=12,
                    uniformtext_mode='hide',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5
                    ),
                    height=750
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### Coded Responses Preview")
                st.dataframe(df_coded.head(20), use_container_width=True, height=600)

    with tab4:  # Exports Tab
        with st.container():
            st.markdown("### 📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Fixed Codeframe Export
                st.markdown("#### Codeframe Export")
                if selected_question in st.session_state.codeframes:
                    try:
                        codeframe = st.session_state.codeframes[selected_question]
                        
                        # Create properly structured dataframe
                        codeframe_data = []
                        for code_num, details in codeframe.items():
                            codeframe_data.append({
                                "Code Number": code_num,
                                "Code Name": details.get("Code Name", ""),
                                "Description": details.get("Description", ""),
                                "Keywords": ", ".join(details.get("Keywords", [])),
                                "Sentiment": details.get("Sentiment", "Neutral")
                            })
                        
                        # Add error code if present in data
                        if selected_question in st.session_state.coded_data:
                            df_coded = st.session_state.coded_data[selected_question]
                            if 999 in pd.Series([item for sublist in df_coded['codes'] for item in sublist]).unique():
                                codeframe_data.append({
                                    "Code Number": 999,
                                    "Code Name": "Processing Error",
                                    "Description": "Failed to code this response",
                                    "Keywords": "",
                                    "Sentiment": "Neutral"
                                })
                        
                        codeframe_df = pd.DataFrame(codeframe_data)
                        
                        # Create downloadable file
                        output_cf_filename = f"codeframe_{selected_question}.xlsx"
                        codeframe_df.to_excel(output_cf_filename, index=False)
                        
                        with open(output_cf_filename, "rb") as f:
                            st.download_button(
                                "💾 Download Codeframe",
                                data=f,
                                file_name=output_cf_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    except Exception as e:
                        st.error(f"Codeframe export error: {str(e)}")

            with col2:
                # Fixed Full Dataset Export
                st.markdown("#### Full Dataset Export")
                if selected_question in st.session_state.coded_data:
                    try:
                        # Get original data and coded results
                        original_df = st.session_state.verbatims.copy()
                        coded_df = st.session_state.coded_data[selected_question].copy()
                        
                        # Clean codes column
                        coded_df['codes'] = coded_df['codes'].apply(
                            lambda x: [c for c in x if isinstance(c, (int, float))]
                        )
                        
                        # Merge with original data
                        final_export = original_df.join(coded_df)
                        
                        # Add individual code columns
                        def safe_get_codes(index):
                            try:
                                return coded_df.loc[index, 'codes']
                            except KeyError:
                                return []
                            
                        max_codes = coded_df['codes'].apply(len).max()
                        code_cols = pd.DataFrame(
                            coded_df['codes'].tolist(),
                            columns=[f'Code_{i+1}' for i in range(max_codes)]
                        )
                        
                        final_export = pd.concat([final_export, code_cols], axis=1)
                        
                        # Generate filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"Coded_{selected_question}_{timestamp}.xlsx"
                        
                        # Create Excel file
                        with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
                            # Main data
                            final_export.to_excel(
                                writer,
                                sheet_name='Coded Responses',
                                index=False
                            )
                            
                            # Codeframe documentation
                            codeframe_df.to_excel(
                                writer,
                                sheet_name='Codeframe',
                                index=False
                            )
                        
                        # Create download button
                        with open(output_filename, 'rb') as f:
                            st.download_button(
                                label='📥 Download Full Dataset(Codeframe+Coded Responses)',
                                data=f,
                                file_name=output_filename,
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                        
                        # Cleanup temporary file
                        os.remove(output_filename)
                        
                    except Exception as e:
                        st.error(f'Export failed: {str(e)}')
                        st.error('Please ensure data consistency between original and coded datasets')

# Footer
st.markdown("---")
current_year = datetime.datetime.now().year
st.markdown(f"<div style='text-align: center; color: #666;'>© {current_year} Survey Open-ended Coding Automation Tool</div>", unsafe_allow_html=True)
