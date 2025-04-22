import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from model import CrowdDensityModel
from utils import preprocess_image, get_density_color

# Configuration
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

# Set page configuration
st.set_page_config(
    page_title="Crowd Density Estimation",
    page_icon="👥",
    layout="wide"
)
tabs = st.tabs(["Image Analysis", "Location-based Analysis"])
# Load model (wrapped in a function to avoid loading at import time)
@st.cache_resource
def load_model():
    """Load the CrowdDensityModel."""
    model = CrowdDensityModel()
    return model

# Initialize the model
model = load_model()

# Main app title and description
st.title("Crowd Density Estimation System")
st.markdown("""
    This application analyzes crowd density using image analysis. 
    Upload an image to classify crowd density as Low, Medium, High, or Extremely High.
""")

# Image Analysis Section
st.header("Image Analysis")
st.markdown("""
Upload an image to analyze crowd density. The system will classify it as:
- **Low**: Few people, sparse distribution
- **Medium**: Moderate gathering, some spacing
- **High**: Dense gathering, limited spacing
- **Extremely High**: Overcrowded, requiring immediate attention
""")

# Upload section
uploaded_file = st.file_uploader("Choose an image file", type=SUPPORTED_FORMATS)

# Display and process the uploaded image
if uploaded_file is not None:
    try:
        # Read the image
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to NumPy array (OpenCV format)
        image = np.array(pil_image)

        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image when the button is clicked
        if st.button("Analyze Crowd Density"):
            with st.spinner("Analyzing crowd density..."):
                # Start timing the processing
                start_time = time.time()

                # Preprocess the image
                preprocessed_img = preprocess_image(image)

                # Make prediction
                density_class, confidence_scores, person_count, density_map = model.predict(preprocessed_img)

                # Calculate processing time
                processing_time = time.time() - start_time

                # Display results
                col1, col2 = st.columns(2)

                # Display analysis results
                with col1:
                    st.subheader("Analysis Results")

                    # Get color based on density class
                    result_color = get_density_color(density_class)

                    # Display the prediction with a colored background
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 5px; background-color: {result_color}; color: white;">
                        <h3 style="margin: 0;">Crowd Density: {density_class}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Confidence Score:** {confidence_scores[['Low', 'Medium', 'High', 'Extremely High'].index(density_class)]:.2f}")
                    st.markdown(f"**Estimated Person Count:** {person_count}")
                    st.markdown(f"**Processing Time:** {processing_time:.4f} seconds")

                    # Display status based on density class
                    if density_class == "Low":
                        st.success("✅ Low crowd density detected")
                    elif density_class == "Medium":
                        st.warning("⚠️ Medium crowd density detected")
                    elif density_class == "High":
                        st.error("🚨 High crowd density detected")
                    else:
                        st.error("🚨 Extremely High crowd density detected! Immediate action required!")
                        st.info(f"**Extremely High Density Count:** {person_count} people detected. Immediate attention required!")

                # Display visualizations
                with col2:
                    st.subheader("Visualization")
                    viz_tabs = st.tabs(["Heatmap Overlay", "Confidence Scores"])

                    # Heatmap overlay visualization
                    with viz_tabs[0]:
                        density_map_normalized = (density_map * 255).astype(np.uint8)
                        density_map_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)

                        # Ensure the heatmap matches the original image dimensions
                        if density_map_colored.shape[:2] != image.shape[:2]:
                            density_map_colored = cv2.resize(density_map_colored, (image.shape[1], image.shape[0]))

                        # Blend the heatmap with the original image
                        blended = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6, density_map_colored, 0.4, 0)
                        st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Density Heatmap", use_column_width=True)

                    # Confidence scores visualization
                    with viz_tabs[1]:
                        labels = ['Low', 'Medium', 'High', 'Extremely High']
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(labels, confidence_scores, color=['green', 'orange', 'red', 'purple'])
                        for bar in bars:
                            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center')
                        ax.set_ylim(0, 1.1)
                        ax.set_ylabel('Confidence Score')
                        ax.set_title('Prediction Confidence by Density Class')
                        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Upload an image to get started!")


# LOCATION-BASED ANALYSIS TAB
with tabs[1]:
    st.header("Location-based Analysis")
    st.markdown("""
    Analyze crowd density based on location data. You can enter a specific location to analyze,
    upload a CSV file with location data, or use the built-in simulation.
    """)
    
    # Option to upload data or use simulation
    data_source = st.radio(
        "Data Source",
        ["Search By Location", "Upload CSV", "Use Simulation"],
        index=0
    )
    
    if data_source == "Search By Location":
        st.subheader("Search By Location")
        st.markdown("""
        Enter a specific location to analyze the crowd density. The system will retrieve data
        for the location and provide real-time analysis.
        """)
        
        # Location search input
        location_input = st.text_input(
            "Enter location name (e.g., City Park, Mall, Train Station)",
            placeholder="Type a location name..."
        )
        
        # Date and time selector
        analysis_date = st.date_input(
            "Select date for analysis",
            value=datetime.now().date()
        )
        
        # Time range slider
        time_range = st.slider(
            "Time range (hours)",
            min_value=0,
            max_value=23,
            value=(8, 18),
            help="Select the time range for analysis in 24-hour format"
        )
        
        if st.button("Analyze Location", key="search_location_btn"):
            if not location_input:
                st.error("Please enter a location name.")
            else:
                with st.spinner(f"Analyzing crowd density at {location_input}..."):
                    try:
                        # Generate sample data for the specific location
                        # In a real app, this would query an API or database
                        data = []
                        
                        # Format date as string
                        date_str = analysis_date.strftime("%Y-%m-%d")
                        
                        # Generate data points for each hour in the selected range
                        for hour in range(time_range[0], time_range[1]+1):
                            # Create timestamp
                            timestamp = f"{date_str} {hour:02d}:00:00"
                            
                            # Generate sample location data based on location type
                            if "park" in location_input.lower():
                                lat, lng = 40.785091, -73.968285
                                base_count = 25 + (15 * abs(hour - 14) / 8)  # Peak at 2pm
                            elif "mall" in location_input.lower():
                                lat, lng = 40.758896, -73.985130
                                base_count = 50 + (40 * abs(hour - 16) / 8)  # Peak at 4pm
                            elif "station" in location_input.lower():
                                lat, lng = 40.750580, -73.993584
                                # Two peaks - morning and evening rush
                                if hour < 12:
                                    base_count = 80 * (1 - abs(hour - 9) / 4)  # Morning peak at 9am
                                else:
                                    base_count = 80 * (1 - abs(hour - 18) / 4)  # Evening peak at 6pm
                            else:
                                lat, lng = 40.712776, -74.005974
                                base_count = 30 + (20 * (1 - abs(hour - 12) / 6))  # Generic peak at noon
                            
                            # Add some randomness
                            count = max(5, int(base_count + np.random.normal(0, base_count * 0.1)))
                            
                            data.append({
                                'location': location_input,
                                'timestamp': timestamp,
                                'latitude': lat,
                                'longitude': lng,
                                'count': count
                            })
                        
                        # Create DataFrame
                        location_data = pd.DataFrame(data)
                        
                        # Display data preview
                        st.subheader("Data Preview")
                        st.dataframe(location_data)
                        
                        # Analyze data for the specific location
                        results, visualizations = analyze_location_data(location_data, model, DENSITY_THRESHOLDS, location_input)
                        
                        if len(results) == 0:
                            st.error(f"No data available for {location_input}.")
                        else:
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Summary statistics
                            avg_count = results['count'].mean()
                            max_count = results['count'].max()
                            min_count = results['count'].min()
                            
                            # Determine the predominant density class
                            density_counts = results['density_class'].value_counts()
                            predominant_class = density_counts.idxmax()
                            
                            # Summary metrics in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Count", f"{avg_count:.1f}")
                            with col2:
                                st.metric("Max Count", max_count)
                            with col3:
                                st.metric("Predominant Density", predominant_class)
                            
                            # Show peak hours
                            peak_hour_data = results.loc[results['count'].idxmax()]
                            peak_time = peak_hour_data['timestamp']
                            st.info(f"Peak crowd time: {peak_time} with {peak_hour_data['count']} people")
                            
                            # Visualizations
                            st.subheader("Crowd Density Over Time")
                            
                            # Time series chart directly with streamlit
                            chart_data = results[['timestamp', 'count']].rename(columns={'count': 'Person Count'})
                            chart_data = chart_data.set_index('timestamp')
                            st.line_chart(chart_data)
                            
                            # Recommendations based on density
                            st.subheader("Recommendations")
                            
                            if predominant_class == "Low":
                                st.success("""
                                ✅ This location generally has low crowd density during the selected period.
                                - Good time for activities requiring space
                                - Minimal crowd management needed
                                - Visitors can expect a comfortable experience
                                """)
                            elif predominant_class == "Medium":
                                st.warning("""
                                ⚠️ This location has moderate crowd density during the selected period.
                                - Consider some crowd management measures
                                - Visitors should expect some waiting times
                                - Plan activities accordingly
                                """)
                            else:
                                st.error("""
                                🚨 This location has high crowd density during the selected period.
                                - Implement crowd control measures
                                - Consider capacity limits or timed entries
                                - Visitors should expect significant waiting times
                                - Consider alternative times for your visit
                                """)
                            
                            # Detailed data table
                            with st.expander("View Detailed Analysis Data"):
                                st.dataframe(results)
                        
                    except Exception as e:
                        st.error(f"Error analyzing location: {str(e)}")
    
    elif data_source == "Upload CSV":
        # CSV file uploader
        uploaded_csv = st.file_uploader(
            "Upload location data (CSV file)", 
            type=["csv"],
            key="csv_uploader"
        )
        
        if uploaded_csv is not None:
            try:
                # Load and validate data
                data = pd.read_csv(uploaded_csv)
                
                # Check for required columns
                required_columns = ['location', 'timestamp', 'latitude', 'longitude']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.markdown(f"""
                    Your CSV file must include the following columns:
                    - `location`: Name or ID of the location
                    - `timestamp`: Time of the measurement (YYYY-MM-DD HH:MM:SS)
                    - `latitude`: Latitude coordinate
                    - `longitude`: Longitude coordinate
                    - `count` (optional): Person count at that location and time
                    """)
                else:
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(data.head())
                    
                    # Option to filter by specific location
                    location_filter = st.selectbox(
                        "Filter by location (optional)",
                        ["All Locations"] + sorted(data['location'].unique().tolist()),
                        index=0
                    )
                    
                    # Analyze data
                    if st.button("Analyze Location Data", key="analyze_location_btn"):
                        with st.spinner("Analyzing location data..."):
                            # If count column doesn't exist, add it based on clustering
                            if 'count' not in data.columns:
                                # Generate counts based on time patterns for demo
                                data['count'] = data.apply(lambda row: 
                                    int(np.random.randint(10, 50) * 
                                        (1 + 0.5 * np.sin(pd.to_datetime(row['timestamp']).hour / 12 * np.pi))), 
                                    axis=1)
                            
                            # Filter by selected location if needed
                            selected_location = None if location_filter == "All Locations" else location_filter
                            
                            # Analyze data
                            results, visualizations = analyze_location_data(data, model, DENSITY_THRESHOLDS, selected_location)
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            if len(results) == 0:
                                st.error("No results found. Please check your data or filters.")
                            else:
                                # Summary by location
                                st.markdown("### Average Crowd Density by Location")
                                location_summary = results.groupby('location').agg({
                                    'count': ['mean', 'max'],
                                    'density_class': lambda x: x.mode()[0]
                                }).reset_index()
                                location_summary.columns = ['Location', 'Avg Count', 'Max Count', 'Typical Density']
                                st.dataframe(location_summary)
                                
                                # Visualizations
                                st.subheader("Visualizations")
                                
                                # Time series chart
                                st.markdown("### Crowd Density Over Time")
                                time_series_fig = create_location_time_series(results, DENSITY_THRESHOLDS)
                                st.pyplot(time_series_fig)
                                
                                # Heatmap (only if multiple locations)
                                if len(results['location'].unique()) > 1:
                                    st.markdown("### Crowd Density Heatmap")
                                    heatmap_fig = create_location_heatmap(results)
                                    st.pyplot(heatmap_fig)
                                
                                # Full results
                                with st.expander("View Detailed Results"):
                                    st.dataframe(results)
            
            except Exception as e:
                st.error(f"Error processing location data: {str(e)}")
        else:
            st.info("👆 Upload a CSV file with location data to get started!")
            st.markdown("""
            ### CSV Format Requirements:
            Your file should include the following columns:
            - `location`: Name or ID of the location
            - `timestamp`: Time of the measurement (YYYY-MM-DD HH:MM:SS)
            - `latitude`: Latitude coordinate
            - `longitude`: Longitude coordinate
            - `count` (optional): Person count at that location and time
            """)
    
    else:  # Simulation mode
        st.subheader("Location Data Simulation")
        st.markdown("""
        Generate simulated location data to demonstrate the crowd density analysis capabilities.
        Adjust the parameters below to customize the simulation.
        """)
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            num_locations = st.slider("Number of Locations", min_value=2, max_value=10, value=4)
            timepoints = st.slider("Number of Timepoints", min_value=5, max_value=24, value=12)
        
        with col2:
            min_count = st.slider("Minimum Person Count", min_value=0, max_value=20, value=5)
            max_count = st.slider("Maximum Person Count", min_value=21, max_value=100, value=50)
        
        # Allow filtering by a specific location
        specific_location = st.checkbox("Filter by a specific location")
        selected_location = None
        
        if specific_location:
            location_options = [f"Location {i+1}" for i in range(num_locations)]
            selected_location = st.selectbox("Select location", location_options)
        
        if st.button("Generate and Analyze Simulated Data", key="sim_analyze_btn"):
            with st.spinner("Generating and analyzing simulated data..."):
                # Generate simulated data
                locations = [f"Location {i+1}" for i in range(num_locations)]
                timestamps = [f"2023-05-15 {i:02d}:00:00" for i in range(timepoints)]
                
                # Create dataframe
                data = []
                for loc in locations:
                    for ts in timestamps:
                        # Generate random count with some pattern (higher in middle of day)
                        hour = int(ts.split()[1].split(":")[0])
                        base_count = min_count + int((max_count - min_count) * 
                                                  (1 - abs(hour - 12) / 12))
                        random_factor = np.random.normal(0, 5)
                        count = max(min_count, min(max_count, int(base_count + random_factor)))
                        
                        data.append({
                            'location': loc,
                            'timestamp': ts,
                            'latitude': 40.7 + np.random.uniform(-0.1, 0.1),
                            'longitude': -74.0 + np.random.uniform(-0.1, 0.1),
                            'count': count
                        })
                
                # Create dataframe
                sim_data = pd.DataFrame(data)
                
                # Display data preview
                st.subheader("Simulated Data Preview")
                st.dataframe(sim_data.head())
                
                # Analyze data
                results, visualizations = analyze_location_data(sim_data, model, DENSITY_THRESHOLDS, selected_location)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Summary by location
                st.markdown("### Average Crowd Density by Location")
                location_summary = results.groupby('location').agg({
                    'count': ['mean', 'max'],
                    'density_class': lambda x: x.mode()[0]
                }).reset_index()
                location_summary.columns = ['Location', 'Avg Count', 'Max Count', 'Typical Density']
                st.dataframe(location_summary)
                
                # Visualizations
                st.subheader("Visualizations")
                
                # Time series chart
                st.markdown("### Crowd Density Over Time")
                time_series_fig = create_location_time_series(results, DENSITY_THRESHOLDS)
                st.pyplot(time_series_fig)
                
                # Heatmap (only if not filtering by location)
                if not specific_location or not selected_location:
                    st.markdown("### Crowd Density Heatmap")
                    heatmap_fig = create_location_heatmap(results)
                    st.pyplot(heatmap_fig)
                
                # Full results
                with st.expander("View Detailed Results"):
                    st.dataframe(results)

# Information about the model
with st.expander("About the Application"):
    st.markdown("""
    ### Crowd Density Estimation System
    
    This system analyzes crowd density using two complementary approaches:
    
    **1. Image Analysis**
    - Upload images to classify crowd density as Low, Medium, or High
    - Uses computer vision techniques to detect and count people
    - Generates heatmaps showing crowd distribution
    
    **2. Location-based Analysis**
    - Analyze crowds by location name, coordinates, or CSV data
    - Track crowd density over time
    - Generate visualizations and recommendations
    
    #### Analysis Features:
    - Edge density and texture analysis for image processing
    - Grid-based crowd detection for dense scenes
    - Time series analysis for location data
    - Classification into 3 density levels (Low, Medium, High)
    """)

# Usage instructions
with st.expander("How to Use"):
    st.markdown("""
    ### Image Analysis
    1. Select the "Image Analysis" tab
    2. Click "Browse files" to upload an image (JPG, JPEG, PNG)
    3. Click "Analyze Crowd Density" 
    4. View the classification results and visualizations
    
    ### Location-based Analysis
    1. Select the "Location-based Analysis" tab
    2. Choose a data source method:
       - **Search By Location**: Enter a location name and date/time
       - **Upload CSV**: Upload a CSV file with location data
       - **Use Simulation**: Generate simulated data for testing
    3. Adjust filters or parameters as needed
    4. Click the appropriate analysis button
    5. View time-based results and recommendations
    """)

if __name__ == "__main__":
    main()