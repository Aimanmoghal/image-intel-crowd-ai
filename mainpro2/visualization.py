import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import io
from PIL import Image

def create_location_time_series(data, density_thresholds):
    """
    Create time series visualization for location data
    
    Args:
        data: DataFrame with location data including timestamp, count, and density_class
        density_thresholds: Dictionary with density classification thresholds
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Check if timestamp is a datetime object for better formatting
    if hasattr(data['timestamp'].iloc[0], 'hour'):
        time_column = 'timestamp'
    else:
        # Try to convert to datetime
        try:
            data['datetime'] = pd.to_datetime(data['timestamp'])
            time_column = 'datetime'
        except:
            time_column = 'timestamp'
    
    # Define a color map for density classes
    color_map = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
    
    # Get unique locations
    locations = data['location'].unique()
    
    # If there's only one location, show density classes with colors
    if len(locations) == 1:
        location = locations[0]
        location_data = data[data['location'] == location]
        
        # Sort by timestamp for a smooth line
        location_data = location_data.sort_values(by=time_column)
        
        # Create scatter points colored by density class
        for density_class in color_map.keys():
            class_data = location_data[location_data['density_class'] == density_class]
            if len(class_data) > 0:
                ax.scatter(
                    class_data[time_column], 
                    class_data['count'], 
                    color=color_map[density_class],
                    label=f"{density_class} Density",
                    alpha=0.7,
                    s=50
                )
        
        # Plot the line connecting all points
        ax.plot(
            location_data[time_column], 
            location_data['count'], 
            marker='o', 
            linestyle='-',
            color='blue',
            alpha=0.5
        )
        
        # Add data labels for point counts
        for i, row in location_data.iterrows():
            ax.annotate(
                f"{row['count']}", 
                (row[time_column], row['count']),
                textcoords="offset points", 
                xytext=(0, 5),
                ha='center',
                fontsize=8
            )
        
        ax.set_title(f'Crowd Density Over Time at {location}')
    else:
        # Multiple locations - use different line styles for each location
        markers = ['o', 's', '^', 'D', 'p', '*', 'x', 'h']
        linestyles = ['-', '--', '-.', ':']
        
        for i, location in enumerate(locations):
            location_data = data[data['location'] == location]
            location_data = location_data.sort_values(by=time_column)
            
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            
            ax.plot(
                location_data[time_column], 
                location_data['count'], 
                marker=marker,
                linestyle=linestyle,
                label=location
            )
        
        ax.set_title('Crowd Density Over Time by Location')
    
    # Format x-axis for better readability
    if hasattr(data[time_column].iloc[0], 'hour'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Person Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add density threshold lines
    ax.axhline(y=density_thresholds['low_max'], color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=density_thresholds['medium_max'], color='red', linestyle='--', alpha=0.5)
    
    # Add threshold labels
    ax.text(
        ax.get_xlim()[1], density_thresholds['low_max'], 
        f'Low-Medium Threshold ({density_thresholds["low_max"]})', 
        ha='right', va='bottom', color='green', fontsize=8
    )
    ax.text(
        ax.get_xlim()[1], density_thresholds['medium_max'], 
        f'Medium-High Threshold ({density_thresholds["medium_max"]})', 
        ha='right', va='bottom', color='red', fontsize=8
    )
    
    plt.tight_layout()
    return fig

def create_location_heatmap(data):
    """
    Create heatmap visualization for location data
    
    Args:
        data: DataFrame with location data
        
    Returns:
        fig: Matplotlib figure object
    """
    # Pivot data to create heatmap format
    try:
        pivot_data = data.pivot(index='location', columns='timestamp', values='count')
    except:
        # If pivot fails (e.g., due to duplicate entries), use alternative approach
        locations = data['location'].unique()
        timestamps = sorted(data['timestamp'].unique())
        pivot_data = np.zeros((len(locations), len(timestamps)))
        
        for i, loc in enumerate(locations):
            for j, ts in enumerate(timestamps):
                subset = data[(data['location'] == loc) & (data['timestamp'] == ts)]
                if len(subset) > 0:
                    pivot_data[i, j] = subset['count'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create heatmap
    heatmap = ax.imshow(pivot_data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Person Count')
    
    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Location')
    ax.set_title('Crowd Density Heatmap by Location and Time')
    
    # Set x and y ticks
    try:
        if isinstance(pivot_data, np.ndarray):
            ax.set_xticks(range(len(timestamps)))
            ax.set_xticklabels([str(ts) for ts in timestamps], rotation=45)
            ax.set_yticks(range(len(locations)))
            ax.set_yticklabels(locations)
        else:
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns, rotation=45)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)
    except:
        # Basic ticks if the above fails
        ax.set_xticks(range(pivot_data.shape[1]))
        ax.set_yticks(range(pivot_data.shape[0]))
    
    plt.tight_layout()
    return fig