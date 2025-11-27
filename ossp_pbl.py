import streamlit as st
import requests  
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split 
import os

# Set page configuration
st.set_page_config(page_title="HealthEdu", page_icon="📚", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 100px;
        color: #4A90E2;
        text-align: center;
        font-weight: bold;
    }
    .menu-title {
        font-size: 50px;
        font-weight: bold;
        color: #4A90E2;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 16px;
        padding: 10px;
        background: white;
    }
    .description {
        font-size: 20px;
        background: rgba(74, 144, 226, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        color: #000000;
    }
    .disclaimer-box h3 {
        color: #000000;
    }
    .disclaimer-box p, .disclaimer-box ul, .disclaimer-box li {
        color: #000000;
    }
    body {
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load datasets
try:
    description_df = pd.read_csv("description.csv")
    diets_df = pd.read_csv("diets.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    training_df = pd.read_csv("training.csv")
    # Try common filename variations
    try:
        symptoms_df = pd.read_csv('symptoms_df.csv')
    except FileNotFoundError:
        symptoms_df = pd.read_csv('symtoms_df.csv')  # Common typo
    medicine_df = pd.read_csv('medications.csv')
except FileNotFoundError as e:
    st.error(f"Error loading dataset: {e}")
    st.write("**Current working directory:**", os.getcwd())
    st.write("**Files in current directory:**")
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    st.write(files)
    st.info("💡 **Tip**: Make sure all CSV files are in the same directory as ossp_pbl.py")
    st.info("The file might be named 'symtoms_df.csv' (with typo) instead of 'symptoms_df.csv'")
    st.stop()

# Global Disclaimer Function
def show_main_disclaimer():
    st.markdown("""
    <div class="disclaimer-box">
        <h3>⚠️ IMPORTANT EDUCATIONAL DISCLAIMER</h3>
        <p><strong>For Educational Purposes Only</strong></p>
        <p>This is NOT medical advice. Always consult qualified healthcare professionals for medical decisions. Never self-diagnose or self-medicate. In emergencies, call emergency services immediately.</p>
    </div>
    """, unsafe_allow_html=True)

# Function to search drug information on Indian government portals
def search_drug_india(medicine_name):
    """Search for drug information using multiple Indian sources"""
    
    # Encode medicine name for URL
    encoded_name = urllib.parse.quote(medicine_name)
    
    # Create search links for Indian medical resources
    links = {
        "CDSCO": f"https://cdsco.gov.in/opencms/opencms/en/Drugs/",
        "IPC": f"https://ipc.gov.in/",
        "1mg": f"https://www.1mg.com/search/all?name={encoded_name}",
        "PharmEasy": f"https://pharmeasy.in/search/all?name={encoded_name}",
        "Drugs.com": f"https://www.drugs.com/search.php?searchterm={encoded_name}"
    }
    
    return links

# Function to get detailed drug information (replace old implementation + add cache + diagnostics)
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_rxterms(term: str):
    """Query RxTerms API; return (status_code, parsed_json or None, error_message)."""
    try:
        url = f"https://clinicaltables.nlm.nih.gov/api/rxterms/v3/search?terms={urllib.parse.quote(term)}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.status_code, r.json(), None
        return r.status_code, None, f"Non-200 status"
    except Exception as e:
        return None, None, str(e)

def normalize_candidates(medicine_name: str):
    """Generate fallback query candidates."""
    base = medicine_name.strip()
    parts = [p.strip() for p in base.replace('(', ' ').replace(')', ' ').replace('/', ' ').split() if p.strip()]
    candidates = [base]
    if len(parts) > 1:
        candidates.append(parts[0])          # first token
        candidates.append(parts[-1])         # last token
    return list(dict.fromkeys(candidates))   # unique preserve order

def get_detailed_drug_info(medicine_name):
    links = search_drug_india(medicine_name)
    api_matches = []
    last_status = None
    last_error = None
    last_url = None
    for candidate in normalize_candidates(medicine_name):
        status, data, err = fetch_rxterms(candidate)
        last_status = status
        last_error = err
        last_url = f"https://clinicaltables.nlm.nih.gov/api/rxterms/v3/search?terms={urllib.parse.quote(candidate)}"
        if data and len(data) > 3 and data[3]:
            api_matches = data[3]
            break
    return {
        'name': medicine_name,
        'links': links,
        'api_data': api_matches[:5] if api_matches else None,
        'status': last_status,
        'error': last_error,
        'url': last_url
    }

# Function to display drug information
def display_drug_information(medicine_name):
    info = get_detailed_drug_info(medicine_name)
    st.markdown(f"### 🔍 Lookup: **{medicine_name}**")
    st.caption(f"Query URL: {info['url']} | Status: {info['status']} | Error: {info['error'] or 'None'}")
    if info['api_data']:
        st.success("Matches:")
        for i, row in enumerate(info['api_data'], 1):
            if isinstance(row, list) and row:
                st.write(f"{i}. {row[0]}")
    else:
        st.info("No direct RxTerms match. Use external resources below.")
    st.markdown("---")
    col_gov, col_db = st.columns(2)
    with col_gov:
        st.markdown("**Government / Official:**")
        st.markdown(f"- [CDSCO]({info['links']['CDSCO']})")
        st.markdown(f"- [IPC]({info['links']['IPC']})")
    with col_db:
        st.markdown("**Databases:**")
        st.markdown(f"- [1mg]({info['links']['1mg']})")
        st.markdown(f"- [PharmEasy]({info['links']['PharmEasy']})")
        st.markdown(f"- [Drugs.com]({info['links']['Drugs.com']})")
    if st.checkbox("Show raw debug data", key=f"dbg_{medicine_name}"):
        st.json(info)
    st.warning("Educational only. Always verify with licensed healthcare providers.")

# Symptom Information System
def symptom_information_system():
    st.markdown('<div class="menu-title">📋 Symptom Information System</div>', unsafe_allow_html=True)
    
    show_main_disclaimer()
    
    st.info("📚 **Educational Tool**: Learn about conditions commonly associated with certain symptoms. This is NOT a diagnosis.")
    
    st.header("Select Symptoms to Learn About Related Conditions")
    symptoms = st.multiselect(
        "Choose symptoms (for educational purposes):",
        training_df.columns[1:-1],
        help="Select symptoms to see which conditions are commonly associated with them"
    )

    if symptoms:
        disease_column = training_df.columns[-1]
        possible_conditions = training_df[training_df[symptoms].sum(axis=1) > 0][disease_column].unique()
        
        if len(possible_conditions) > 0:
            st.warning("⚠️ **Educational Information Only**: Many symptoms can be associated with multiple conditions. Professional medical evaluation is required for accurate diagnosis.")
            st.write("**Conditions commonly associated with these symptoms:**", list(possible_conditions))
            
            # ML Model Section
            st.subheader("📊 Pattern Matching Analysis (Educational)")
            st.info("This is a simplified educational model using machine learning for demonstration purposes only.")
            
            symptom_columns = training_df.columns[1:-1]
            x = training_df[symptom_columns]
            y = training_df[disease_column]
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            input_data = {symptom: (1 if symptom in symptoms else 0) for symptom in symptom_columns}
            input_df = pd.DataFrame([input_data])
            predicted_condition = model.predict(input_df)[0]
            
            st.write(f"**Pattern matching suggests**: {predicted_condition}")
            st.error("⚠️ **Critical Reminder**: This is a simplified educational model. Real medical diagnosis requires:")
            st.write("- Physical examination by a doctor")
            st.write("- Medical history review")
            st.write("- Laboratory tests and imaging")
            st.write("- Professional medical expertise")
            
            # Show educational information
            show_condition_information(predicted_condition)
            
            # Consult Doctor Section
            st.subheader("🏥 Next Steps: Consult a Healthcare Provider")
            st.write("""
            **You should see a doctor if:**
            - Symptoms persist for more than a few days
            - Symptoms worsen or become severe
            - You experience high fever, severe pain, or difficulty breathing
            - You have concerns about your health
            
            **How to Find Medical Help:**
            - Contact your primary care physician
            - Visit an urgent care clinic
            - Go to the emergency room for serious symptoms
            - Use telehealth services for initial consultation
            """)
            
        else:
            st.write("No matching patterns found in the educational database.")
    else:
        st.write("Please select symptoms to see educational information.")

# Show condition information
def show_condition_information(condition):
    st.markdown("---")
    st.subheader(f"📖 Educational Information About: {condition}")
    st.warning("⚠️ This is general educational information. Your specific situation may differ. Consult a doctor for personalized advice.")
    
    # Description
    description = description_df[description_df['Disease'] == condition]['Description'].values
    if len(description) > 0:
        st.write("**General Information:**")
        st.write(description[0])
    
    # Diet Information
    diet = diets_df[diets_df['Disease'] == condition]['Diet'].values
    if len(diet) > 0:
        st.write("**Dietary Considerations (General guidance):**")
        st.info(diet[0])
        st.caption("Consult a nutritionist or doctor for personalized dietary advice.")
    
    # Precautions
    precautions = precautions_df[precautions_df['Disease'] == condition].iloc[:, 1:].values.flatten()
    if len(precautions) > 0:
        st.write("**General Precautions (Educational):**")
        st.write(", ".join([str(p) for p in precautions if pd.notna(p)]))
        st.caption("Always follow your healthcare provider's specific recommendations.")
    
    # Workout Information
    workout = workout_df[workout_df['disease'] == condition]['workout'].values
    if len(workout) > 0:
        st.write("**Physical Activity Considerations:**")
        st.info(workout[0])
        st.caption("Consult your doctor before starting any exercise program.")

# Medication Education Tool
def medication_education_tool():
    st.markdown('<div class="menu-title">💊 Medication Education Tool</div>', unsafe_allow_html=True)
    
    show_main_disclaimer()
    
    st.info("📚 **Learning Tool**: Understand which medications are commonly associated with certain conditions. This is NOT a prescription or recommendation.")
    
    # Clean the symptom data
    all_symptoms_raw = symptoms_df.iloc[:, :-1].stack().dropna().unique()
    all_symptoms = [str(symptom).strip().lower() for symptom in all_symptoms_raw if symptom]
    all_symptoms = [symptom for symptom in all_symptoms if symptom and symptom.replace(' ', '').isalpha()]
    all_symptoms = sorted(set(all_symptoms))

    st.subheader("Educational Symptom-to-Treatment Information")
    st.write("Click symptoms to learn about commonly prescribed medication classes:")

    # Initialize session state
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    if 'show_medicines' not in st.session_state:
        st.session_state.show_medicines = False
    if 'matching_medicines' not in st.session_state:
        st.session_state.matching_medicines = []
    if 'possible_conditions' not in st.session_state:
        st.session_state.possible_conditions = []

    # Display symptoms as buttons
    num_columns = 5
    symptom_chunks = [all_symptoms[i:i + num_columns] for i in range(0, len(all_symptoms), num_columns)]

    for chunk in symptom_chunks:
        cols = st.columns(num_columns)
        for col, symptom in zip(cols, chunk):
            if col.button(symptom, key=f"symptom_{symptom}"):
                if symptom in st.session_state.selected_symptoms:
                    st.session_state.selected_symptoms.remove(symptom)
                else:
                    st.session_state.selected_symptoms.append(symptom)
                st.session_state.show_medicines = False
                # no st.rerun() — avoid full script rerun on each click

    st.write("**Selected Symptoms:**", st.session_state.selected_symptoms if st.session_state.selected_symptoms else "None")

    if st.button("Learn About Treatment Options"):
        if not st.session_state.selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        # Map Symptoms to Conditions
        possible_conditions = []
        disease_col = symptoms_df.columns[-1]
        
        for _, row in symptoms_df.iterrows():
            disease = row[disease_col]
            symptoms_in_row = row[:-1].dropna().apply(lambda x: str(x).strip().lower()).tolist()
            if any(symptom in symptoms_in_row for symptom in st.session_state.selected_symptoms):
                possible_conditions.append(disease)

        # Convert all conditions to strings and remove NaN values
        st.session_state.possible_conditions = [str(c) for c in list(set(possible_conditions)) if pd.notna(c)]
        
        if possible_conditions:
            st.info(f"**Conditions commonly associated with these symptoms (Educational):** {', '.join(st.session_state.possible_conditions)}")

            # Map Conditions to Medications
            matching_medicines = []
            for condition in possible_conditions:
                # Skip if condition is NaN or not a string
                if pd.isna(condition):
                    continue
                    
                condition_medicines = medicine_df[medicine_df['Disease'].str.lower() == str(condition).lower()]['Medication'].values
                
                if len(condition_medicines) > 0:
                    med_value = condition_medicines[0]
                    
                    if isinstance(med_value, str):
                        try:
                            import ast
                            if med_value.startswith('['):
                                med_list = ast.literal_eval(med_value)
                                matching_medicines.extend(med_list)
                            else:
                                matching_medicines.extend([m.strip() for m in med_value.split(',')])
                        except:
                            matching_medicines.append(med_value)
                    elif isinstance(med_value, list):
                        matching_medicines.extend(med_value)
                    else:
                        matching_medicines.append(str(med_value))

            # Clean and convert all medicines to strings
            st.session_state.matching_medicines = list(set([
                str(m).strip() for m in matching_medicines 
                if m and pd.notna(m) and str(m).lower() not in ['nan', 'none', '']
            ]))
            st.session_state.show_medicines = True
        else:
            st.warning("No matching conditions found in the educational database.")
            st.session_state.show_medicines = False
    
    # Display medicine information
    if st.session_state.show_medicines and st.session_state.matching_medicines:
        # ...existing code (listing medicines)...
        st.markdown("---")
        st.subheader("🔍 Look Up Medication Details")
        selected_medicine = st.selectbox(
            "Choose a medication:",
            ["Select a medication..."] + st.session_state.matching_medicines,
            key="med_lookup_selectbox"
        )
        # Persist selection + trigger
        if 'edu_lookup_trigger' not in st.session_state:
            st.session_state.edu_lookup_trigger = False
            st.session_state.edu_lookup_med = None
        if st.button("Lookup Medication Details", key="lookup_med_btn"):
            if selected_medicine != "Select a medication...":
                st.session_state.edu_lookup_trigger = True
                st.session_state.edu_lookup_med = selected_medicine
            else:
                st.warning("Select a medication first.")
        if st.session_state.get('edu_lookup_trigger') and st.session_state.get('edu_lookup_med'):
            display_drug_information(st.session_state.edu_lookup_med)
        st.warning("""
        **NEVER take any medication without:**
        - A proper medical diagnosis from a licensed healthcare provider
        - A valid prescription from your doctor
        - Understanding the dosage, timing, and potential interactions
        - Discussing your complete medical history with your doctor
        
        **Taking medication without medical supervision can be dangerous and potentially fatal.**
        """)
        
        st.subheader("🏥 What to Do Next")
        st.write("""
        1. **Schedule an appointment** with your healthcare provider
        2. **Describe your symptoms** in detail to your doctor
        3. **Get a proper diagnosis** through examination and tests if needed
        4. **Receive a prescription** if medication is appropriate
        5. **Follow instructions** from your doctor and pharmacist exactly
        """)

# Medication Verification Learning Tool
def medication_verification_tool():
    st.markdown('<div class="menu-title">🔍 Medication Verification Learning Tool</div>', unsafe_allow_html=True)
    
    show_main_disclaimer()
    
    st.info("📚 **Educational Tool**: Learn whether a medication is typically associated with a specific condition. This does NOT verify if YOUR prescription is correct.")
    
    st.subheader("Learn About Medication-Condition Associations")
    
    conditions = sorted(medicine_df['Disease'].unique().tolist())
    
    selected_condition = st.selectbox(
        "Select a condition (for educational purposes):", 
        conditions,
        key="condition_selectbox"
    )
    medication_input = st.text_area(
        "Enter medication name(s) to learn about (comma separated):",
        key="med_input_textarea"
    )
    
    if st.button("Check Educational Database"):
        if not medication_input:
            st.warning("Please enter at least one medication name.")
            return
        
        entered_medications = [med.strip() for med in medication_input.split(',')]
        
        typical_medications_raw = medicine_df[medicine_df['Disease'] == selected_condition]['Medication'].values
        
        if len(typical_medications_raw) > 0:
            typical_medications = []
            med_value = typical_medications_raw[0]
            
            if isinstance(med_value, str):
                try:
                    import ast
                    if med_value.startswith('['):
                        typical_medications = ast.literal_eval(med_value)
                    else:
                        typical_medications = [m.strip() for m in med_value.split(',')]
                except:
                    typical_medications = [med_value]
            elif isinstance(med_value, list):
                typical_medications = med_value
            else:
                typical_medications = [str(med_value)]
        else:
            typical_medications = []
        
        matched = [med for med in entered_medications if med in typical_medications]
        not_matched = [med for med in entered_medications if med not in typical_medications]
        
        st.subheader("Educational Information:")
        
        if matched:
            st.success(f"**Typically associated**: {', '.join(matched)}")
            st.write(f"These medications are commonly prescribed for {selected_condition} according to our educational database.")
        
        if not_matched:
            st.info(f"**Not typically associated**: {', '.join(not_matched)}")
            st.write(f"These medications are not commonly listed for {selected_condition} in our educational database.")
        
        if typical_medications:
            st.write(f"**Medications commonly associated with {selected_condition}:**")
            for idx, med in enumerate(typical_medications, 1):
                st.write(f"{idx}. {med}")
            
            st.markdown("---")
            st.subheader("🔍 Look Up Individual Medications")
            
            all_entered_meds = matched + not_matched
            filtered = [m for m in all_entered_meds if m]
            lookup_med = st.selectbox(
                "Select a medication to look up online:",
                ["Select a medication..."] + filtered,
                key="verification_lookup_selectbox"
            )
            if 'ver_lookup_trigger' not in st.session_state:
                st.session_state.ver_lookup_trigger = False
                st.session_state.ver_lookup_med = None
            if st.button("Lookup Medication Details (Verification)", key="verification_lookup_btn"):
                if lookup_med != "Select a medication...":
                    st.session_state.ver_lookup_trigger = True
                    st.session_state.ver_lookup_med = lookup_med
                else:
                    st.warning("Select a medication first.")
            if st.session_state.get('ver_lookup_trigger') and st.session_state.get('ver_lookup_med'):
                display_drug_information(st.session_state.ver_lookup_med)

# Health Data Visualization
def health_data_visualization():
    st.markdown('<div class="menu-title">📊 Health Data Visualization</div>', unsafe_allow_html=True)
    
    st.info("📚 **Educational Visualization**: Explore patterns in health data for learning purposes.")
    
    try:
        data = training_df.copy()
        disease_col = data.columns[-1]
        symptom_data = data.drop(columns=[disease_col])
        disease_counts = data[disease_col].value_counts()
        correlation_matrix = symptom_data.corr()

        visualization_type = st.selectbox(
            "Choose a visualization type:",
            ["Condition Frequency", "Distribution Chart", "Symptom Scatter Plot"]
        )

        if visualization_type == "Condition Frequency":
            st.subheader("Frequency of Conditions in Educational Dataset")
            st.bar_chart(disease_counts)
            st.caption("This shows the distribution of conditions in the training dataset.")

        elif visualization_type == "Distribution Chart":
            st.subheader("Condition Distribution (Educational)")
            fig, ax = plt.subplots()
            ax.pie(disease_counts.head(10), labels=disease_counts.head(10).index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            st.caption("Top 10 conditions by frequency in the educational dataset.")

        elif visualization_type == "Symptom Scatter Plot":
            st.subheader("Symptom Relationship Analysis (Educational)")
            symptom_cols = [col for col in data.columns if col != disease_col]
            if len(symptom_cols) >= 2:
                fig = px.scatter(data, x=symptom_cols[0], y=symptom_cols[1], color=disease_col, 
                               title=f"Example: Relationship Between {symptom_cols[0]} and {symptom_cols[1]}")
                st.plotly_chart(fig)
                st.caption("This shows how symptoms might cluster by condition in the dataset.")
            else:
                st.write("Not enough symptom columns available for scatter plot.")
                
    except Exception as e:
        st.error(f"Error in visualization: {e}")

# Professional Resources
def show_professional_resources():
    st.subheader("🏥 Professional Medical Resources")
    st.write("""
    **For accurate medical information and care, consult:**
    
    **Immediate Medical Help:**
    - Emergency Services: Call your local emergency number (911 in US, 112 in Europe, etc.)
    - Poison Control: Contact your local poison control center
    
    **Regular Medical Care:**
    - Your Primary Care Physician
    - Local Hospitals and Clinics
    - Urgent Care Centers
    - Telehealth Services
    
    **Verified Online Medical Information:**
    - World Health Organization (WHO): [who.int](https://www.who.int)
    - Centers for Disease Control (CDC): [cdc.gov](https://www.cdc.gov)
    - Mayo Clinic: [mayoclinic.org](https://www.mayoclinic.org)
    - MedlinePlus: [medlineplus.gov](https://medlineplus.gov)
    - National Institutes of Health (NIH): [nih.gov](https://www.nih.gov)
    
    **Remember:** Online information, including this app, should NEVER replace professional medical advice.
    """)

# Main Application
st.markdown('<div class="main-title">HealthEdu 📚</div>', unsafe_allow_html=True)
st.write("### Your Health Education Companion")

# Sidebar Menu
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.info("**Educational Platform**: Learn about health topics to have informed conversations with healthcare providers.")

menu_options = [
    "🏠 Home",
    "📋 Symptom Information System",
    "💊 Medication Education",
    "🔍 Medication Verification Learning",
    "📊 Health Data Visualization",
    "🏥 Professional Resources"
]

choice = st.sidebar.radio("Select an Option:", menu_options)

# Page Routing
if choice == "🏠 Home":
    st.markdown("### Welcome to HealthEdu!")
    
    show_main_disclaimer()
    
    st.markdown("""
    <div class="description">
    <h4>About HealthEdu</h4>
    <p>HealthEdu is an educational platform designed to help you learn about health topics, 
    symptoms, conditions, and treatment approaches. This tool helps you understand health 
    information so you can have more informed conversations with your healthcare providers.</p>
    
    <p><strong>What HealthEdu Offers:</strong></p>
    <ul>
        <li>Learn about symptoms and commonly associated conditions</li>
        <li>Understand medication classes typically prescribed for different conditions</li>
        <li>Explore general health information and wellness practices</li>
        <li>Visualize health data patterns for educational purposes</li>
    </ul>
    
    <p><strong>What HealthEdu is NOT:</strong></p>
    <ul>
        <li>NOT a substitute for professional medical advice</li>
        <li>NOT a diagnostic tool</li>
        <li>NOT a prescription service</li>
        <li>NOT a replacement for visiting a doctor</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Why Choose HealthEdu?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎓 Educational Focus")
        st.write("Learn about health topics in an accessible, easy-to-understand format")
    
    with col2:
        st.markdown("#### 🔒 Ethical & Safe")
        st.write("Clear disclaimers and guidance to consult healthcare professionals")
    
    with col3:
        st.markdown("#### 📊 Data-Driven")
        st.write("Information based on medical datasets and established health knowledge")
    
    # Try to load image if it exists
    image_path = 'What-is-Healthcare-Health-Care-healthcare-nt-sickcare-2783.webp'
    if os.path.exists(image_path):
        st.image(image_path, width=1200, 
                 caption="Healthcare Education - Learn to understand your health better")
    else:
        st.info("💡 Place an image named 'What-is-Healthcare-Health-Care-healthcare-nt-sickcare-2783.webp' in the same directory to display it here.")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use This Platform")
    st.write("""
    1. **Explore the Symptom Information System** - Learn about conditions associated with symptoms
    2. **Medication Education** - Understand medication classes and their general uses
    3. **Verify Your Knowledge** - Check if medications are typically associated with conditions
    4. **Visualize Health Data** - See patterns in health information
    5. **Always Consult Professionals** - Use this knowledge to have better conversations with your doctor
    """)
    
    show_professional_resources()

elif choice == "📋 Symptom Information System":
    symptom_information_system()

elif choice == "💊 Medication Education":
    medication_education_tool()

elif choice == "🔍 Medication Verification Learning":
    medication_verification_tool()

elif choice == "📊 Health Data Visualization":
    health_data_visualization()

elif choice == "🏥 Professional Resources":
    st.markdown('<div class="menu-title">🏥 Professional Medical Resources</div>', unsafe_allow_html=True)
    show_main_disclaimer()
    show_professional_resources()
    
    # Add Indian-specific resources
    st.markdown("---")
    st.subheader("🇮🇳 India-Specific Medical Resources")
    st.write("""
    **Government Health Portals:**
    - [Ministry of Health & Family Welfare](https://www.mohfw.gov.in/)
    - [CDSCO - Drug Regulator](https://cdsco.gov.in/)
    - [Indian Pharmacopoeia Commission](https://ipc.gov.in/)
    - [National Health Portal](https://www.nhp.gov.in/)
    - [Ayushman Bharat](https://pmjay.gov.in/)
    
    **Emergency Services:**
    - National Emergency Number: **112**
    - Ambulance: **102 / 108** (varies by state)
    - Medical Helpline: **104**
    
    **Verified Indian Medical Databases:**
    - [1mg - Medicine Information](https://www.1mg.com/)
    - [PharmEasy](https://pharmeasy.in/)
    - [Apollo Pharmacy](https://www.apollopharmacy.in/)
    
    **Find Doctors & Hospitals:**
    - [Practo](https://www.practo.com/)
    - [DocPrime](https://www.docprime.com/)
    - [Lybrate](https://www.lybrate.com/)
    """)


