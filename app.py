import streamlit as st
import numpy as np
from PIL import Image
from utils.numeric_augmentation import (
    apply_numeric_augmentation, 
    calculate_central_tendencies, 
    display_metrics_tiles, 
    plot_distributions
)

st.set_page_config(layout="wide")
st.title("🧠 ML with Small Data")

st.header("Numeric Data Augmentation with Central Tendencies")

raw_input = st.text_area("Enter comma-separated numbers", "2,3,3,4,5,5,4,3,2,1")
try:
    data = np.array([float(i.strip()) for i in raw_input.split(",") if i.strip()])
except:
    st.warning("Invalid input. Please enter comma-separated numbers.")
    st.stop()

method = st.selectbox("Choose Numeric Augmentation", [
    # Original methods
    "1. Gaussian Noise", 
    "2. Gaussian on Sorted Data", 
    "3. Jitter",
    "4. Linear Interpolation", 
    "5. Quadratic Interpolation",
    "6. Cubic Spline Interpolation", 
    "7. KDE Sampling", 
    "8. SMOTE (1D Simulated)",
    
    # Bootstrap-based methods
    "9. Bootstrap Resampling",
    "10. Block Bootstrap",
    "11. Parametric Bootstrap",
    
    # Statistical distribution methods
    "12. Uniform Noise",
    "13. Laplace Noise", 
    "14. Poisson Noise",
    "15. Exponential Scaling",
    
    # Pattern-based methods
    "16. Magnitude Warping",
    "17. Time Warping",
    "18. Window Slicing",
    "19. Permutation",
    
    # Mathematical transformations
    "20. Log Transform + Noise",
    "21. Box-Cox Transform + Noise",
    "22. Z-Score Normalization + Noise",
    "23. Min-Max Scaling + Noise",
    
    # Advanced synthetic methods
    "24. VAE (Simplified)",
    "25. GAN (Simplified)",
    "26. Mixup (Linear Combination)",
    "27. CutMix (Segment Mixing)",
    
    # Ensemble methods
    "28. Multi-Method Ensemble",
    "29. Weighted Combination",
    "30. Random Method Selection"
])

augmented, synthetic = apply_numeric_augmentation(data, method)

original_metrics = calculate_central_tendencies(data)
synthetic_metrics = calculate_central_tendencies(synthetic) if len(synthetic) > 0 else {}
augmented_metrics = calculate_central_tendencies(augmented)

st.subheader("📊 Central Tendencies Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    display_metrics_tiles(original_metrics, "Original Data", "#3498db")

with col2:
    if len(synthetic) > 0:
        display_metrics_tiles(synthetic_metrics, "Synthetic Data", "#e74c3c")
    else:
        st.markdown("### Synthetic Data")
        st.info("No synthetic data generated for this method")

with col3:
    display_metrics_tiles(augmented_metrics, "Augmented Data", "#2ecc71")

st.subheader("📈 Distribution Visualizations")
plot_distributions(data, synthetic, augmented, method)

st.subheader("📋 Data Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Original Data:**")
    st.write(data.tolist())

with col2:
    st.write("**Synthetic Data:**")
    if len(synthetic) > 0:
        st.write(synthetic.round(3).tolist())
    else:
        st.write("No synthetic data generated")

with col3:
    st.write("**Augmented Data:**")
    st.write(augmented.round(3).tolist())