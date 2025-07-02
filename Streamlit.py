import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime, timedelta
import numpy as np
import re

# Page configuration
st.set_page_config(
    page_title="Adaptive Training Dashboard",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    /* Main header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(90deg, #00c851, #007e33);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Warning message styling */
    .warning-message {
        background: linear-gradient(90deg, #ffbb33, #ff8800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .insight-card h4 {
        margin-top: 0;
        font-size: 1.2rem;
    }
    
    /* Deployment info styling */
    .deployment-info {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Deployment helper functions
def get_google_credentials():
    """Get Google credentials from Streamlit secrets or file upload"""
    try:
        # Try to get from Streamlit Cloud secrets first
        if hasattr(st, 'secrets') and 'connections' in st.secrets and 'gsheets' in st.secrets['connections']:
            return dict(st.secrets['connections']['gsheets'])
        else:
            return None
    except Exception:
        return None

def create_connection_interface():
    """Create interface for Google Sheets connection"""
    st.markdown("#### 📊 Google Sheets Connection")
    
    # Check if running on Streamlit Cloud with secrets
    cloud_credentials = get_google_credentials()
    
    if cloud_credentials:
        st.success("✅ Google Sheets credentials configured via Streamlit Cloud secrets")
        
        # Google Sheets URL input
        sheets_url = st.text_input(
            "Google Sheets URL:",
            placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit",
            help="Paste your Google Sheets URL here"
        )
        
        if sheets_url:
            connector = GoogleSheetsConnector(cloud_credentials, sheets_url)
            
            if st.button("🔗 Connect to Google Sheets"):
                with st.spinner("Connecting to Google Sheets..."):
                    if connector.connect():
                        st.success("✅ Connected successfully!")
                        st.session_state['connected'] = True
                        st.session_state['connector'] = connector
                    else:
                        st.error("❌ Connection failed")
        
        return sheets_url
    
    else:
        st.info("💡 Running locally - upload credentials file")
        
        # Google Sheets URL input
        sheets_url = st.text_input(
            "Google Sheets URL:",
            placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit",
            help="Paste your Google Sheets URL here"
        )
        
        # Credentials upload for local development
        st.write("Upload Google Service Account JSON:")
        uploaded_file = st.file_uploader(
            "Choose credentials file",
            type=['json'],
            help="Upload your Google Service Account JSON file"
        )
        
        if uploaded_file and sheets_url:
            try:
                credentials = json.load(uploaded_file)
                connector = GoogleSheetsConnector(credentials, sheets_url)
                
                if st.button("🔗 Connect to Google Sheets"):
                    with st.spinner("Connecting to Google Sheets..."):
                        if connector.connect():
                            st.success("✅ Connected successfully!")
                            st.session_state['connected'] = True
                            st.session_state['connector'] = connector
                        else:
                            st.error("❌ Connection failed")
                            
            except Exception as e:
                st.error(f"Error loading credentials: {str(e)}")
        
        return sheets_url

class TrainingAnalytics:
    """Advanced analytics and insights for training data"""
    
    def __init__(self, df):
        self.df = df
        
    def get_volume_trends(self):
        """Calculate volume trends and progression rates"""
        weekly_volume = self.df.groupby('week')['volume_calculated'].sum().reset_index()
        
        if len(weekly_volume) > 1:
            # Calculate week-to-week change
            weekly_volume['volume_change'] = weekly_volume['volume_calculated'].pct_change() * 100
            weekly_volume['volume_change_abs'] = weekly_volume['volume_calculated'].diff()
        else:
            weekly_volume['volume_change'] = 0
            weekly_volume['volume_change_abs'] = 0
            
        return weekly_volume
    
    def get_muscle_group_balance(self):
        """Analyze muscle group balance and recommendations"""
        muscle_volume = self.df.groupby('muscleGroup')['volume_calculated'].sum().reset_index()
        total_volume = muscle_volume['volume_calculated'].sum()
        muscle_volume['percentage'] = (muscle_volume['volume_calculated'] / total_volume * 100).round(1)
        
        # Ideal percentages (these can be customized)
        ideal_percentages = {
            'Chest': 15, 'Back': 20, 'Shoulders': 15, 'Arms': 15,
            'Legs': 25, 'Core': 10
        }
        
        muscle_volume['ideal_percentage'] = muscle_volume['muscleGroup'].map(ideal_percentages).fillna(10)
        muscle_volume['balance_score'] = muscle_volume['percentage'] - muscle_volume['ideal_percentage']
        
        return muscle_volume
    
    def get_exercise_rotation(self):
        """Analyze exercise variety and rotation patterns"""
        exercise_weeks = self.df.groupby('exercise')['week'].apply(list).reset_index()
        exercise_weeks['weeks_used'] = exercise_weeks['week'].apply(len)
        exercise_weeks['weeks_list'] = exercise_weeks['week'].apply(lambda x: sorted(x))
        
        # Calculate variety metrics
        total_weeks = self.df['week'].nunique()
        total_exercises = len(exercise_weeks)
        avg_exercises_per_week = len(self.df) / total_weeks if total_weeks > 0 else 0
        
        return {
            'exercise_rotation': exercise_weeks,
            'total_exercises': total_exercises,
            'avg_exercises_per_week': round(avg_exercises_per_week, 1),
            'variety_score': round(total_exercises / total_weeks if total_weeks > 0 else 0, 1)
        }
    
    def get_training_insights(self):
        """Generate intelligent training insights"""
        insights = []
        
        # Volume progression insight
        volume_trends = self.get_volume_trends()
        if len(volume_trends) > 1:
            latest_change = volume_trends['volume_change'].iloc[-1]
            if latest_change > 10:
                insights.append({
                    'title': '📈 Strong Volume Progression',
                    'message': f'Your training volume increased by {latest_change:.1f}% this week. Great progressive overload!',
                    'type': 'success'
                })
            elif latest_change < -10:
                insights.append({
                    'title': '📉 Volume Decrease Detected',
                    'message': f'Volume decreased by {abs(latest_change):.1f}% this week. Consider if this was planned deload.',
                    'type': 'warning'
                })
        
        # Muscle group balance insight
        balance_data = self.get_muscle_group_balance()
        imbalanced = balance_data[abs(balance_data['balance_score']) > 10]
        if not imbalanced.empty:
            worst_imbalance = imbalanced.loc[imbalanced['balance_score'].abs().idxmax()]
            if worst_imbalance['balance_score'] > 0:
                insights.append({
                    'title': f'⚖️ {worst_imbalance["muscleGroup"]} Overemphasis',
                    'message': f'{worst_imbalance["muscleGroup"]} is {worst_imbalance["balance_score"]:.1f}% above recommended volume.',
                    'type': 'info'
                })
            else:
                insights.append({
                    'title': f'⚖️ {worst_imbalance["muscleGroup"]} Undertraining',
                    'message': f'{worst_imbalance["muscleGroup"]} is {abs(worst_imbalance["balance_score"]):.1f}% below recommended volume.',
                    'type': 'warning'
                })
        
        # Exercise variety insight
        rotation_data = self.get_exercise_rotation()
        if rotation_data['variety_score'] < 2:
            insights.append({
                'title': '🔄 Low Exercise Variety',
                'message': f'Consider adding more exercise variety. Current: {rotation_data["variety_score"]} exercises per week.',
                'type': 'info'
            })
        elif rotation_data['variety_score'] > 6:
            insights.append({
                'title': '🎯 Excellent Exercise Variety',
                'message': f'Great exercise diversity with {rotation_data["variety_score"]} exercises per week!',
                'type': 'success'
            })
        
        return insights

class TrainingDataProcessor:
    """Enhanced data processing with advanced calculations"""
    
    def __init__(self, data):
        self.data = data
        self.df = self.process_data()
        self.analytics = TrainingAnalytics(self.df)
    
    def process_data(self):
        """Convert raw data to pandas DataFrame and add calculated columns"""
        if not self.data:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.data)
        
        # Calculate numeric volume (handle range strings like "3-4")
        df['sets_numeric'] = df['sets'].apply(self.extract_numeric_value)
        df['reps_numeric'] = df['reps'].apply(self.extract_numeric_value)
        df['volume_calculated'] = df['sets_numeric'] * df['reps_numeric']
        
        # Add training intensity categories
        df['volume_category'] = pd.cut(df['volume_calculated'], 
                                     bins=[0, 20, 40, 60, float('inf')],
                                     labels=['Low', 'Moderate', 'High', 'Very High'])
        
        # Add exercise type classification
        df['exercise_type'] = df['exercise'].apply(self.classify_exercise_type)
        
        return df
    
    def extract_numeric_value(self, value_str):
        """Extract numeric value from strings like '3-4' or '8-12'"""
        if pd.isna(value_str) or value_str == '':
            return 0
        
        value_str = str(value_str).strip()
        
        # Skip if it's clearly a header or non-numeric text
        if value_str.lower() in ['sets', 'reps', 'rest', 'exercise', 'muscle group', 'muscle', '']:
            return 0
        
        try:
            return float(value_str)
        except:
            try:
                if '-' in value_str:
                    parts = value_str.split('-')
                    if len(parts) == 2:
                        try:
                            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                        except:
                            return 0
                else:
                    numbers = re.findall(r'\d+(?:\.\d+)?', value_str)
                    if numbers:
                        return float(numbers[0])
                    else:
                        return 0
            except:
                return 0
    
    def classify_exercise_type(self, exercise_name):
        """Classify exercises into compound vs isolation"""
        compound_keywords = ['squat', 'deadlift', 'press', 'pull-up', 'chin-up', 'row', 'lunge']
        exercise_lower = exercise_name.lower()
        
        for keyword in compound_keywords:
            if keyword in exercise_lower:
                return 'Compound'
        return 'Isolation'
    
    def get_weekly_volume(self):
        """Calculate total volume by week"""
        return self.df.groupby('week')['volume_calculated'].sum().reset_index()
    
    def get_muscle_group_distribution(self):
        """Calculate volume distribution by muscle group"""
        return self.df.groupby('muscleGroup')['volume_calculated'].sum().reset_index()
    
    def get_exercise_frequency(self):
        """Calculate exercise frequency across all weeks"""
        return self.df.groupby('exercise').size().reset_index(name='frequency')
    
    def get_weekly_muscle_breakdown(self):
        """Get volume by week and muscle group for heatmap"""
        return self.df.groupby(['week', 'muscleGroup'])['volume_calculated'].sum().reset_index()
    
    def get_exercise_type_analysis(self):
        """Analyze compound vs isolation exercise distribution"""
        return self.df.groupby('exercise_type')['volume_calculated'].sum().reset_index()

class GoogleSheetsConnector:
    """Enhanced Google Sheets connector with better error handling"""
    
    def __init__(self, credentials_dict, spreadsheet_url, show_status=True):
        self.credentials_dict = credentials_dict
        self.spreadsheet_url = spreadsheet_url
        self.client = None
        self.worksheet = None
        self.show_status = show_status
        
    def connect(self):
        """Establish connection to Google Sheets"""
        try:
            scope = [
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly"
            ]
            
            creds = Credentials.from_service_account_info(
                self.credentials_dict, 
                scopes=scope
            )
            
            self.client = gspread.authorize(creds)
            self.worksheet = self.client.open_by_url(self.spreadsheet_url)
            return True
            
        except Exception as e:
            st.error(f"Failed to connect to Google Sheets: {str(e)}")
            return False
    
    def extract_training_data(self):
        """Extract training data from Week 1-8 sheets with enhanced debugging"""
        all_data = []
        
        for week_num in range(1, 9):
            try:
                sheet_name = f"Week {week_num}"
                worksheet = self.worksheet.worksheet(sheet_name)
                
                # Get raw values to handle headers manually
                all_values = worksheet.get_all_values()
                
                if not all_values:
                    continue
                
                # Find the header row and clean it
                header_row = None
                data_start_row = 0
                
                # Look for the row with "Muscle Group" or similar headers
                for i, row in enumerate(all_values):
                    row_text = ' '.join([str(cell).lower() for cell in row])
                    if any(keyword in row_text for keyword in ['muscle', 'exercise', 'sets', 'reps']):
                        header_row = row
                        data_start_row = i + 1
                        break
                
                if header_row is None:
                    continue
                
                # Clean and standardize headers
                cleaned_headers = []
                for header in header_row:
                    clean_header = header.strip().lower().replace(' ', '').replace('-', '').replace('_', '')
                    
                    # Map common variations to standard names
                    header_mapping = {
                        'musclegroup': 'Muscle Group',
                        'muscle': 'Muscle Group',
                        'musclegrp': 'Muscle Group',
                        'exercise': 'Exercise',
                        'exercises': 'Exercise',
                        'sets': 'Sets',
                        'set': 'Sets',
                        'reps': 'Reps',
                        'rep': 'Reps',
                        'repetitions': 'Reps',
                        'rest': 'Rest',
                        'resttime': 'Rest',
                        'restperiod': 'Rest'
                    }
                    
                    standard_header = header_mapping.get(clean_header, header.strip() if header.strip() else f'Column_{len(cleaned_headers)}')
                    cleaned_headers.append(standard_header)
                
                # Process data rows
                for row_idx in range(data_start_row, len(all_values)):
                    row = all_values[row_idx]
                    
                    # Skip empty rows
                    if not any(cell.strip() for cell in row):
                        continue
                    
                    # Create row dictionary
                    row_data = {}
                    for col_idx, value in enumerate(row):
                        if col_idx < len(cleaned_headers):
                            row_data[cleaned_headers[col_idx]] = value.strip()
                    
                    # Check if this row has the essential data
                    muscle_group = row_data.get('Muscle Group', '').strip()
                    exercise = row_data.get('Exercise', '').strip()
                    
                    # Skip rows that are clearly headers or invalid data
                    header_keywords = ['muscle group', 'muscle', 'exercise', 'sets', 'reps', 'rest']
                    if (muscle_group.lower() in header_keywords or 
                        exercise.lower() in header_keywords or
                        muscle_group == '' or exercise == ''):
                        continue
                    
                    if muscle_group and exercise:
                        exercise_data = {
                            'week': week_num,
                            'weekName': f"Week {week_num}",
                            'muscleGroup': muscle_group,
                            'exercise': exercise,
                            'sets': row_data.get('Sets', ''),
                            'reps': row_data.get('Reps', ''),
                            'rest': row_data.get('Rest', ''),
                            'estimatedVolume': self.calculate_volume(
                                row_data.get('Sets', ''), 
                                row_data.get('Reps', '')
                            )
                        }
                        all_data.append(exercise_data)
                
                exercises_this_week = len([d for d in all_data if d['week'] == week_num])
                if exercises_this_week > 0 and self.show_status:
                    st.success(f"✅ Successfully read {exercises_this_week} exercises from {sheet_name}")
                        
            except Exception as e:
                if "does not exist" not in str(e) and self.show_status:  # Don't show errors for missing weeks
                    st.warning(f"Could not read {sheet_name}: {str(e)}")
                
        return all_data
    
    def calculate_volume(self, sets_str, reps_str):
        """Calculate estimated volume from sets and reps strings"""
        try:
            def get_mid_value(range_str):
                if not range_str or str(range_str).strip() == '':
                    return 0
                
                range_str = str(range_str).strip()
                
                # Skip header text
                if range_str.lower() in ['sets', 'reps', 'rest', 'exercise', 'muscle group']:
                    return 0
                
                try:
                    if '-' in range_str:
                        parts = range_str.split('-')
                        if len(parts) == 2:
                            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                    else:
                        return float(range_str)
                except:
                    numbers = re.findall(r'\d+(?:\.\d+)?', range_str)
                    if numbers:
                        return float(numbers[0])
                    return 0
            
            sets = get_mid_value(sets_str)
            reps = get_mid_value(reps_str)
            return sets * reps
            
        except Exception as e:
            return 0

def load_sample_data():
    """Load sample data for demo purposes"""
    sample_data = [
        {"week": 1, "weekName": "Week 1", "muscleGroup": "Chest", "exercise": "Bench Press", "sets": "3-4", "reps": "8-12", "rest": "2-3 min", "estimatedVolume": 40},
        {"week": 1, "weekName": "Week 1", "muscleGroup": "Back", "exercise": "Pull-ups", "sets": "3", "reps": "6-10", "rest": "2-3 min", "estimatedVolume": 24},
        {"week": 1, "weekName": "Week 1", "muscleGroup": "Legs", "exercise": "Squats", "sets": "4", "reps": "8-10", "rest": "3-4 min", "estimatedVolume": 36},
        {"week": 1, "weekName": "Week 1", "muscleGroup": "Shoulders", "exercise": "Shoulder Press", "sets": "3", "reps": "8-12", "rest": "2-3 min", "estimatedVolume": 30},
        {"week": 2, "weekName": "Week 2", "muscleGroup": "Chest", "exercise": "Incline Press", "sets": "3-4", "reps": "8-12", "rest": "2-3 min", "estimatedVolume": 40},
        {"week": 2, "weekName": "Week 2", "muscleGroup": "Back", "exercise": "Rows", "sets": "3-4", "reps": "8-12", "rest": "2-3 min", "estimatedVolume": 35},
        {"week": 2, "weekName": "Week 2", "muscleGroup": "Legs", "exercise": "Deadlifts", "sets": "3", "reps": "5-8", "rest": "3-4 min", "estimatedVolume": 19.5},
        {"week": 3, "weekName": "Week 3", "muscleGroup": "Chest", "exercise": "Dumbbell Press", "sets": "3-4", "reps": "10-12", "rest": "2-3 min", "estimatedVolume": 38.5},
        {"week": 3, "weekName": "Week 3", "muscleGroup": "Shoulders", "exercise": "Lateral Raises", "sets": "3", "reps": "12-15", "rest": "1-2 min", "estimatedVolume": 40.5},
        {"week": 4, "weekName": "Week 4", "muscleGroup": "Back", "exercise": "Lat Pulldown", "sets": "3-4", "reps": "10-12", "rest": "2-3 min", "estimatedVolume": 38.5},
    ]
    return sample_data

def create_enhanced_charts(processor):
    """Create advanced interactive charts"""
    charts = {}
    
    if processor.df.empty:
        return charts
    
    # 1. Enhanced Weekly Volume Progression with Trend Line
    weekly_data = processor.get_weekly_volume()
    charts['weekly_volume'] = go.Figure()
    
    # Main volume line
    charts['weekly_volume'].add_trace(go.Scatter(
        x=weekly_data['week'],
        y=weekly_data['volume_calculated'],
        mode='lines+markers',
        name='Training Volume',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8, color='#2980b9')
    ))
    
    # Add trend line if multiple weeks
    if len(weekly_data) > 1:
        z = np.polyfit(weekly_data['week'], weekly_data['volume_calculated'], 1)
        p = np.poly1d(z)
        charts['weekly_volume'].add_trace(go.Scatter(
            x=weekly_data['week'],
            y=p(weekly_data['week']),
            mode='lines',
            name='Trend',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            opacity=0.7
        ))
    
    charts['weekly_volume'].update_layout(
        title="📈 Weekly Training Volume Progression",
        xaxis_title="Week",
        yaxis_title="Total Volume (Sets × Reps)",
        template="plotly_white",
        hovermode='x unified'
    )
    
    # 2. Enhanced Muscle Group Distribution with percentages
    muscle_data = processor.get_muscle_group_distribution()
    total_volume = muscle_data['volume_calculated'].sum()
    muscle_data['percentage'] = (muscle_data['volume_calculated'] / total_volume * 100).round(1)
    
    charts['muscle_distribution'] = px.pie(
        muscle_data,
        values='volume_calculated',
        names='muscleGroup',
        title="🎯 Training Volume by Muscle Group",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    charts['muscle_distribution'].update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Volume: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    # 3. Advanced Muscle Group Heatmap
    heatmap_data = processor.get_weekly_muscle_breakdown()
    if not heatmap_data.empty:
        pivot_data = heatmap_data.pivot(index='muscleGroup', columns='week', values='volume_calculated').fillna(0)
        
        charts['muscle_heatmap'] = px.imshow(
            pivot_data,
            title="🔥 Weekly Muscle Group Volume Heatmap",
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(x="Week", y="Muscle Group", color="Volume")
        )
        
        charts['muscle_heatmap'].update_layout(
            xaxis_title="Week",
            yaxis_title="Muscle Group"
        )
    
    # 4. Exercise Type Analysis (Compound vs Isolation)
    exercise_type_data = processor.get_exercise_type_analysis()
    if not exercise_type_data.empty:
        charts['exercise_types'] = px.bar(
            exercise_type_data,
            x='exercise_type',
            y='volume_calculated',
            title="💪 Compound vs Isolation Exercise Volume",
            color='exercise_type',
            color_discrete_map={'Compound': '#2ecc71', 'Isolation': '#f39c12'}
        )
        
        charts['exercise_types'].update_layout(
            xaxis_title="Exercise Type",
            yaxis_title="Total Volume",
            showlegend=False
        )
    
    # 5. Top Exercises Frequency
    freq_data = processor.get_exercise_frequency()
    if not freq_data.empty:
        top_exercises = freq_data.nlargest(10, 'frequency')
        
        charts['exercise_frequency'] = px.bar(
            top_exercises,
            x='frequency',
            y='exercise',
            orientation='h',
            title="🏆 Top 10 Most Frequent Exercises",
            color='frequency',
            color_continuous_scale='Blues'
        )
        
        charts['exercise_frequency'].update_layout(
            xaxis_title="Frequency (Times Used)",
            yaxis_title="Exercise"
        )
    
    # 6. Volume Categories Distribution
    if 'volume_category' in processor.df.columns:
        volume_cat_data = processor.df.groupby('volume_category').size().reset_index(name='count')
        
        charts['volume_categories'] = px.pie(
            volume_cat_data,
            values='count',
            names='volume_category',
            title="📊 Exercise Volume Distribution",
            color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        )
    
    return charts

def display_training_insights(processor):
    """Display intelligent training insights"""
    if processor.df.empty:
        return
        
    insights = processor.analytics.get_training_insights()
    
    if insights:
        st.markdown('<h3 class="section-header">🧠 Training Insights & Recommendations</h3>', unsafe_allow_html=True)
        
        for insight in insights:
            if insight['type'] == 'success':
                icon = "✅"
                bg_class = "success-message"
            elif insight['type'] == 'warning':
                icon = "⚠️"
                bg_class = "warning-message"
            else:
                icon = "💡"
                bg_class = "insight-card"
            
            st.markdown(f"""
                <div class="{bg_class}">
                    <h4>{icon} {insight['title']}</h4>
                    <p>{insight['message']}</p>
                </div>
            """, unsafe_allow_html=True)

def display_deployment_info():
    """Display deployment information and sharing instructions"""
    if not st.session_state.get('show_deployment_info', False):
        return
    
    st.markdown("""
        <div class="deployment-info">
            <h3>🚀 Deployment Success!</h3>
            <p><strong>Your dashboard is now live and shareable!</strong></p>
            <p>📱 <strong>Share this URL with your business partner:</strong><br>
            https://your-app-name.streamlit.app</p>
            <p>🔗 <strong>Features available:</strong></p>
            <ul>
                <li>✅ Real-time Google Sheets sync</li>
                <li>✅ AI-powered training insights</li>
                <li>✅ Interactive charts and analytics</li>
                <li>✅ Professional reporting tools</li>
                <li>✅ Mobile-responsive design</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def main():
    """Enhanced main Streamlit application"""
    
    # Initialize presentation mode in session state
    if 'presentation_mode' not in st.session_state:
        st.session_state.presentation_mode = True
    
    # Header
    st.markdown('<h1 class="dashboard-header">🏋️ Adaptive Training Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Business presentation welcome message
    if st.session_state.presentation_mode:
        st.markdown("""
            <div class="deployment-info">
                <h3>🚀 Professional Training Analytics Platform</h3>
                <p><strong>Real-time insights powered by advanced algorithms</strong></p>
                <p>📊 <strong>Live Data Visualization</strong> • 🧠 <strong>AI-Powered Recommendations</strong> • 📈 <strong>Performance Tracking</strong></p>
                <p>Use the filters in the sidebar to explore different aspects of the training program.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display deployment info if applicable
    display_deployment_info()
    
    # Sidebar with presentation mode toggle
    with st.sidebar:
        # Presentation mode toggle at the top
        st.session_state.presentation_mode = st.toggle("🎯 Presentation Mode", 
                                                       value=st.session_state.presentation_mode, 
                                                       help="Hide technical options for clean business presentation")
        
        if not st.session_state.presentation_mode:
            st.markdown("### ⚙️ Technical Configuration")
            
            # Data source selection
            data_source = st.radio(
                "Select Data Source:",
                ["Sample Data (Demo)", "Google Sheets (Live)"],
                help="Choose between demo data or your live Google Sheets"
            )
            
            if data_source == "Google Sheets (Live)":
                sheets_url = create_connection_interface()
        else:
            # Auto-select appropriate data source for presentation
            if 'connected' in st.session_state and st.session_state['connected']:
                data_source = "Google Sheets (Live)"
                sheets_url = None
            else:
                data_source = "Sample Data (Demo)"
                sheets_url = None
        
        # Enhanced Filters section (always visible but clean in presentation mode)
        if st.session_state.presentation_mode:
            st.markdown("### 🎛️ Training Filters")
        else:
            st.markdown("#### 🎛️ Advanced Filters")
        
        # Week range filter
        week_range = st.slider(
            "Week Range:",
            min_value=1,
            max_value=8,
            value=(1, 8),
            step=1,
            help="Select which weeks to include in analysis"
        )
        
        # Muscle group filter
        if data_source == "Sample Data (Demo)":
            available_muscle_groups = ['All', 'Chest', 'Back', 'Legs', 'Shoulders', 'Arms']
        else:
            available_muscle_groups = ['All']  # Will be populated with real data
        
        selected_muscle_groups = st.multiselect(
            "Muscle Groups:",
            options=available_muscle_groups,
            default=['All'],
            help="Filter by specific muscle groups"
        )
        
        if not st.session_state.presentation_mode:
            # Volume threshold filter (hidden in presentation mode)
            min_volume = st.slider(
                "Minimum Exercise Volume:",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="Filter exercises below this volume"
            )
            
            # Advanced options
            st.markdown("#### 🔧 Advanced Options")
            
            show_trends = st.checkbox("📈 Show Trend Lines", value=True)
            show_insights = st.checkbox("🧠 Show AI Insights", value=True)
            debug_mode = st.checkbox("🔍 Debug Mode", help="Show technical details")
            
            # Refresh data button
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        else:
            # Set defaults for presentation mode
            min_volume = 0
            show_trends = True
            show_insights = True
            debug_mode = False
            
            # Add business-focused info
            st.markdown("---")
            st.markdown("""
                ### 💼 Business Dashboard
                **Real-time training analytics**
                - 📊 Live data visualization
                - 🧠 AI-powered insights
                - 📈 Progress tracking
                - ⚖️ Balance analysis
                """)
            
            if data_source == "Sample Data (Demo)":
                st.info("🎯 **Demo Mode**: Showing sample training data to demonstrate capabilities.")
            else:
                st.success("✅ **Live Data**: Connected to real training program.")
    
    # Main content area
    try:
        # Load data based on source
        if data_source == "Sample Data (Demo)":
            data = load_sample_data()
            if not st.session_state.presentation_mode:
                st.info("📋 Using sample data for demonstration. Connect to Google Sheets for live data.")
        else:
            if 'connected' in st.session_state and st.session_state['connected']:
                with st.spinner("Loading data from Google Sheets..."):
                    if debug_mode:
                        st.subheader("🔍 Debug: Sheet Structure")
                        try:
                            worksheet = st.session_state['connector'].worksheet.worksheet("Week 1")
                            all_values = worksheet.get_all_values()
                            
                            st.write("**First 10 rows of Week 1:**")
                            debug_df = pd.DataFrame(all_values[:10])
                            st.dataframe(debug_df, use_container_width=True)
                            
                            st.write("**Available sheet names:**")
                            sheet_names = [ws.title for ws in st.session_state['connector'].worksheet.worksheets()]
                            st.write(sheet_names)
                            
                        except Exception as e:
                            st.error(f"Debug error: {str(e)}")
                    
                    data = st.session_state['connector'].extract_training_data()
                if not data:
                    if not st.session_state.presentation_mode:
                        st.warning("No data found in Google Sheets")
                    data = load_sample_data()
            else:
                if not st.session_state.presentation_mode:
                    st.warning("Please connect to Google Sheets first")
                data = []
        
        if not data:
            if st.session_state.presentation_mode:
                st.info("🎯 **Demo Mode**: Switch to 'Sample Data' in the sidebar to explore the dashboard capabilities.")
            else:
                st.error("No training data available")
            return
        
        # Filter data by week range and other criteria
        filtered_data = [d for d in data if week_range[0] <= d['week'] <= week_range[1]]
        
        # Apply muscle group filter
        if 'All' not in selected_muscle_groups and selected_muscle_groups:
            filtered_data = [d for d in filtered_data if d['muscleGroup'] in selected_muscle_groups]
        
        # Apply volume filter
        if min_volume > 0:
            filtered_data = [d for d in filtered_data if d.get('estimatedVolume', 0) >= min_volume]
        
        # Process data
        processor = TrainingDataProcessor(filtered_data)
        
        if processor.df.empty:
            if st.session_state.presentation_mode:
                st.info("🎛️ **No data matches current filters.** Try adjusting the filters in the sidebar.")
            else:
                st.warning("No data matches your current filters")
            return
        
        # Enhanced Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_volume = processor.df['volume_calculated'].sum()
            st.metric("🎯 Total Volume", f"{total_volume:.0f}", 
                     help="Total training volume (sets × reps)")
        
        with col2:
            total_exercises = len(processor.df['exercise'].unique())
            st.metric("💪 Unique Exercises", total_exercises,
                     help="Number of different exercises")
        
        with col3:
            muscle_groups = len(processor.df['muscleGroup'].unique())
            st.metric("🏋️ Muscle Groups", muscle_groups,
                     help="Number of muscle groups trained")
        
        with col4:
            weeks_trained = len(processor.df['week'].unique())
            st.metric("📅 Weeks Trained", weeks_trained,
                     help="Number of weeks with training data")
        
        # Training Insights Section
        if show_insights:
            display_training_insights(processor)
        
        # Enhanced Charts section
        st.markdown('<h3 class="section-header">📊 Advanced Training Analytics</h3>', unsafe_allow_html=True)
        
        charts = create_enhanced_charts(processor)
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'weekly_volume' in charts:
                st.plotly_chart(charts['weekly_volume'], use_container_width=True)
        
        with col2:
            if 'muscle_distribution' in charts:
                st.plotly_chart(charts['muscle_distribution'], use_container_width=True)
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'muscle_heatmap' in charts:
                st.plotly_chart(charts['muscle_heatmap'], use_container_width=True)
        
        with col2:
            if 'exercise_types' in charts:
                st.plotly_chart(charts['exercise_types'], use_container_width=True)
        
        # Third row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'exercise_frequency' in charts:
                st.plotly_chart(charts['exercise_frequency'], use_container_width=True)
        
        with col2:
            if 'volume_categories' in charts:
                st.plotly_chart(charts['volume_categories'], use_container_width=True)
        
        # Advanced Analytics Section
        st.markdown('<h3 class="section-header">🔬 Advanced Performance Analysis</h3>', unsafe_allow_html=True)
        
        # Volume trends analysis
        volume_trends = processor.analytics.get_volume_trends()
        if len(volume_trends) > 1 and 'volume_change' in volume_trends.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Volume Progression Rate")
                latest_change = volume_trends['volume_change'].iloc[-1]
                if not pd.isna(latest_change):
                    change_color = "🟢" if latest_change > 0 else "🔴" if latest_change < 0 else "🟡"
                    st.metric("Week-to-Week Change", f"{latest_change:.1f}%", delta=f"{latest_change:.1f}%")
                
            with col2:
                st.subheader("⚖️ Muscle Group Balance")
                balance_data = processor.analytics.get_muscle_group_balance()
                if not balance_data.empty:
                    most_trained = balance_data.loc[balance_data['percentage'].idxmax(), 'muscleGroup']
                    highest_percentage = balance_data['percentage'].max()
                    st.metric("Most Trained", most_trained, f"{highest_percentage:.1f}%")
        
        # Exercise rotation analysis
        rotation_data = processor.analytics.get_exercise_rotation()
        if rotation_data['total_exercises'] > 0:
            st.subheader("🔄 Exercise Variety Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Exercises", rotation_data['total_exercises'])
            with col2:
                st.metric("Avg per Week", rotation_data['avg_exercises_per_week'])
            with col3:
                st.metric("Variety Score", rotation_data['variety_score'])
        
        # Data table section with enhanced filtering
        with st.expander("📋 Detailed Training Log", expanded=False):
            # Additional table filters
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", ['week', 'muscleGroup', 'exercise', 'volume_calculated'])
            with col2:
                sort_order = st.selectbox("Order:", ['Ascending', 'Descending'])
            
            # Sort dataframe
            ascending = sort_order == 'Ascending'
            display_df = processor.df.sort_values(sort_by, ascending=ascending)
            
            st.dataframe(
                display_df[['week', 'muscleGroup', 'exercise', 'sets', 'reps', 'volume_calculated']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'week': 'Week',
                    'muscleGroup': 'Muscle Group',
                    'exercise': 'Exercise',
                    'sets': 'Sets',
                    'reps': 'Reps',
                    'volume_calculated': st.column_config.NumberColumn('Volume', format="%.0f")
                }
            )
        
        # Enhanced Download section
        st.markdown('<h3 class="section-header">⬇️ Export & Reports</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = processor.df.to_csv(index=False)
            st.download_button(
                label="📄 Download Training Data (CSV)",
                data=csv,
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = processor.df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download Data (JSON)",
                data=json_data,
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col3:
            # Generate summary report
            summary_report = f"""# Training Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
- Total Volume: {processor.df['volume_calculated'].sum():.0f}
- Unique Exercises: {len(processor.df['exercise'].unique())}
- Muscle Groups: {len(processor.df['muscleGroup'].unique())}
- Weeks Trained: {len(processor.df['week'].unique())}

## Top Exercises by Volume
{processor.df.groupby('exercise')['volume_calculated'].sum().sort_values(ascending=False).head().to_string()}

## Volume by Muscle Group
{processor.df.groupby('muscleGroup')['volume_calculated'].sum().sort_values(ascending=False).to_string()}
"""
            st.download_button(
                label="📊 Download Summary Report",
                data=summary_report,
                file_name=f"training_summary_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        if st.session_state.presentation_mode:
            st.error("Unable to load data. Please try refreshing the page.")
        else:
            st.error(f"An error occurred: {str(e)}")
            if debug_mode:
                st.exception(e)
            st.info("Please check your data source and try again.")

if __name__ == "__main__":
    main()
