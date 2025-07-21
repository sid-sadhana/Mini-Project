import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde, uniform, laplace, poisson, expon, boxcox
from scipy.interpolate import CubicSpline
from scipy.special import inv_boxcox
import pandas as pd
import random

def calculate_central_tendencies(data):
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        return {}
    
    return {
        'Mean': np.mean(data_clean),
        'Median': np.median(data_clean),
        'Mode': float(pd.Series(data_clean).mode().iloc[0]) if len(pd.Series(data_clean).mode()) > 0 else np.nan,
        'Std Dev': np.std(data_clean),
        'Variance': np.var(data_clean),
        'Min': np.min(data_clean),
        'Max': np.max(data_clean),
        'Range': np.max(data_clean) - np.min(data_clean),
        'Skewness': pd.Series(data_clean).skew(),
        'Kurtosis': pd.Series(data_clean).kurtosis(),
        'Count': len(data_clean)
    }

def display_metrics_tiles(metrics, title, color):
    st.markdown(f"### {title}")
    
    cols = st.columns(4)
    metrics_items = list(metrics.items())
    
    for i, (metric, value) in enumerate(metrics_items):
        col_idx = i % 4
        with cols[col_idx]:
            if not np.isnan(value):
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}, {color}dd);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 5px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0; color: white; font-size: 14px;">{metric}</h4>
                    <h3 style="margin: 5px 0 0 0; color: white;">{value:.3f}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #cccccc, #999999);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 5px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0; color: white; font-size: 14px;">{metric}</h4>
                    <h3 style="margin: 5px 0 0 0; color: white;">N/A</h3>
                </div>
                """, unsafe_allow_html=True)

def apply_numeric_augmentation(data, method):
    
    if method == "1. Gaussian Noise":
        mu, sigma = np.mean(data), np.std(data)
        noise = np.random.normal(mu, sigma, len(data))
        synthetic = np.round(noise)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "2. Gaussian on Sorted Data":
        sorted_data = np.sort(data)
        mu, sigma = np.mean(sorted_data), np.std(sorted_data)
        noise = np.random.normal(mu, sigma, len(sorted_data))
        synthetic = np.round(noise)
        return np.concatenate([sorted_data, synthetic]), synthetic

    elif method == "3. Jitter":
        jittered = data + np.random.normal(0, 0.2, len(data))
        synthetic = np.round(jittered)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "4. Linear Interpolation":
        if len(data) < 2: 
            return data, np.array([])
        synthetic = (data[:-1] + data[1:]) / 2
        return np.concatenate([data, synthetic]), synthetic

    elif method == "5. Quadratic Interpolation":
        if len(data) < 3: 
            return data, np.array([])
        synthetic = []
        for i in range(len(data) - 2):
            x = [i, i+1, i+2]
            y = data[i:i+3]
            coefs = np.polyfit(x, y, 2)
            f = np.poly1d(coefs)
            synthetic.append(f(i + 1.5))
        synthetic = np.array(synthetic)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "6. Cubic Spline Interpolation":
        if len(data) < 4: 
            return data, np.array([])
        x = np.arange(len(data))
        cs = CubicSpline(x, data)
        x_new = np.linspace(0, len(data)-1, 2*len(data))
        synthetic = cs(x_new)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "7. KDE Sampling":
        kde = gaussian_kde(data)
        synthetic = kde.resample(len(data)).flatten()
        return np.concatenate([data, synthetic]), synthetic

    elif method == "8. SMOTE (1D Simulated)":
        if len(data) < 2: 
            return data, np.array([])
        data_ = data.reshape(-1, 1)
        k = min(5, len(data)-1)
        nn = NearestNeighbors(n_neighbors=k).fit(data_)
        neighbors = nn.kneighbors(data_, return_distance=False)
        synthetic = []
        for i in range(len(data_)):
            for j in neighbors[i][1:]:
                lam = np.random.rand()
                synthetic.append(data_[i] + lam * (data_[j] - data_[i]))
        synthetic = np.array(synthetic).flatten()
        return np.concatenate([data.flatten(), synthetic]), synthetic

    elif method == "9. Bootstrap Resampling":
        synthetic = np.random.choice(data, size=len(data), replace=True)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "10. Block Bootstrap":
        if len(data) < 3:
            return data, np.array([])
        block_size = max(2, len(data) // 3)
        synthetic = []
        while len(synthetic) < len(data):
            start_idx = np.random.randint(0, len(data) - block_size + 1)
            block = data[start_idx:start_idx + block_size]
            synthetic.extend(block)
        synthetic = np.array(synthetic[:len(data)])
        return np.concatenate([data, synthetic]), synthetic

    elif method == "11. Parametric Bootstrap":
        mu, sigma = np.mean(data), np.std(data)
        synthetic = np.random.normal(mu, sigma, len(data))
        return np.concatenate([data, synthetic]), synthetic

    elif method == "12. Uniform Noise":
        min_val, max_val = np.min(data), np.max(data)
        range_val = max_val - min_val
        synthetic = np.random.uniform(min_val - 0.1*range_val, max_val + 0.1*range_val, len(data))
        return np.concatenate([data, synthetic]), synthetic

    elif method == "13. Laplace Noise":
        loc, scale = np.mean(data), np.std(data)
        synthetic = np.random.laplace(loc, scale, len(data))
        return np.concatenate([data, synthetic]), synthetic

    elif method == "14. Poisson Noise":
        data_pos = np.abs(data) + 1
        synthetic = []
        for val in data_pos:
            synthetic.append(np.random.poisson(val))
        synthetic = np.array(synthetic)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "15. Exponential Scaling":
        scale_factors = np.random.exponential(1.0, len(data))
        synthetic = data * scale_factors
        return np.concatenate([data, synthetic]), synthetic

    elif method == "16. Magnitude Warping":
        window_size = max(2, len(data) // 4)
        synthetic = data.copy()
        for _ in range(len(data) // window_size):
            start_idx = np.random.randint(0, len(data) - window_size + 1)
            scale_factor = np.random.uniform(0.5, 2.0)
            synthetic[start_idx:start_idx + window_size] *= scale_factor
        return np.concatenate([data, synthetic]), synthetic

    elif method == "17. Time Warping":
        if len(data) < 4:
            return data, np.array([])
        x_original = np.arange(len(data))
        warp_factor = np.random.uniform(0.8, 1.2)
        x_warped = np.linspace(0, len(data)-1, int(len(data) * warp_factor))
        synthetic = np.interp(x_original, x_warped * (len(data)-1) / x_warped[-1], 
                            np.interp(x_warped, x_original, data))
        return np.concatenate([data, synthetic]), synthetic

    elif method == "18. Window Slicing":
        if len(data) < 4:
            return data, np.array([])
        window_size = max(2, len(data) // 3)
        synthetic = []
        for _ in range(len(data) // window_size):
            start_idx = np.random.randint(0, len(data) - window_size + 1)
            window = data[start_idx:start_idx + window_size]
            synthetic.extend(window)
        synthetic = np.array(synthetic[:len(data)])
        return np.concatenate([data, synthetic]), synthetic

    elif method == "19. Permutation":
        synthetic = np.random.permutation(data)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "20. Log Transform + Noise":
        data_pos = np.abs(data) + 1e-6
        log_data = np.log(data_pos)
        noise = np.random.normal(0, np.std(log_data) * 0.1, len(data))
        synthetic = np.exp(log_data + noise)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "21. Box-Cox Transform + Noise":
        try:
            data_pos = data - np.min(data) + 1
            transformed, lambda_param = boxcox(data_pos)
            noise = np.random.normal(0, np.std(transformed) * 0.1, len(data))
            synthetic = inv_boxcox(transformed + noise, lambda_param)
            synthetic = synthetic + np.min(data) - 1
        except:
            synthetic = data + np.random.normal(0, np.std(data) * 0.1, len(data))
        return np.concatenate([data, synthetic]), synthetic

    elif method == "22. Z-Score Normalization + Noise":
        z_scores = (data - np.mean(data)) / np.std(data)
        noise = np.random.normal(0, 0.1, len(data))
        synthetic = (z_scores + noise) * np.std(data) + np.mean(data)
        return np.concatenate([data, synthetic]), synthetic

    elif method == "23. Min-Max Scaling + Noise":
        min_val, max_val = np.min(data), np.max(data)
        normalized = (data - min_val) / (max_val - min_val)
        noise = np.random.uniform(-0.1, 0.1, len(data))
        synthetic = np.clip(normalized + noise, 0, 1)
        synthetic = synthetic * (max_val - min_val) + min_val
        return np.concatenate([data, synthetic]), synthetic

    elif method == "24. VAE (Simplified)":
        mu, sigma = np.mean(data), np.std(data)
        latent = np.random.normal(0, 1, len(data))
        synthetic = mu + sigma * latent
        return np.concatenate([data, synthetic]), synthetic


    elif method == "25. GAN (Simplified)":
        synthetic = []
        for _ in range(len(data)):
            z = np.random.normal(0, 1)
            mean, std = np.mean(data), np.std(data)
            val = z * std + mean 
            val += np.random.normal(0, std * 0.05) 
            synthetic.append(val)
        synthetic = np.array(synthetic)
        return np.concatenate([data, synthetic]), synthetic


    elif method == "26. Mixup (Linear Combination)":
        synthetic = []
        for _ in range(len(data)):
            i, j = np.random.choice(len(data), 2, replace=True)
            lam = np.random.beta(0.4, 0.4)
            synthetic.append(lam * data[i] + (1 - lam) * data[j])
        synthetic = np.array(synthetic)
        return np.concatenate([data, synthetic]), synthetic


    elif method == "27. CutMix (Segment Mixing)":
        if len(data) < 4:
            return data, np.array([])

        synthetic = []
        for _ in range(len(data)):
            a, b = np.random.choice(len(data), 2, replace=True)
            lam = np.random.uniform(0.2, 0.8)
            cut_point = int(len(data) * lam)

            new_sample = np.concatenate([
                data[a:cut_point],
                data[b:cut_point:]
            ])
            synthetic.append(np.mean(new_sample))  
        synthetic = np.array(synthetic)
        return np.concatenate([data, synthetic]), synthetic


    elif method == "28. Multi-Method Ensemble":
        methods = ["1. Gaussian Noise", "7. KDE Sampling", "9. Bootstrap Resampling"]
        all_synthetic = []
        for m in methods:
            _, synth = apply_numeric_augmentation(data, m)
            if len(synth) > 0:
                all_synthetic.extend(synth[:len(data)//3])
        synthetic = np.array(all_synthetic[:len(data)])
        return np.concatenate([data, synthetic]), synthetic

    elif method == "29. Weighted Combination":
        _, synth1 = apply_numeric_augmentation(data, "1. Gaussian Noise")
        _, synth2 = apply_numeric_augmentation(data, "7. KDE Sampling")
        _, synth3 = apply_numeric_augmentation(data, "9. Bootstrap Resampling")
        
        weights = [0.4, 0.3, 0.3]
        synthetic = []
        min_len = min(len(synth1), len(synth2), len(synth3))
        for i in range(min_len):
            combined = (weights[0] * synth1[i] + weights[1] * synth2[i] + weights[2] * synth3[i])
            synthetic.append(combined)
        synthetic = np.array(synthetic[:len(data)])
        return np.concatenate([data, synthetic]), synthetic

    elif method == "30. Random Method Selection":
        available_methods = [
            "1. Gaussian Noise", "3. Jitter", "7. KDE Sampling", 
            "9. Bootstrap Resampling", "12. Uniform Noise", "26. Mixup (Linear Combination)"
        ]
        selected_method = np.random.choice(available_methods)
        return apply_numeric_augmentation(data, selected_method)

    else:
        return data, np.array([])

def plot_distributions(original, synthetic, augmented, method_name):

    def create_histogram_data(data, label):
        if len(data) == 0:
            return pd.DataFrame()
        
        hist, bin_edges = np.histogram(data, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        df = pd.DataFrame({
            'Value': [f"{edge:.1f}" for edge in bin_centers],
            'Frequency': hist
        })
        return df
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Original Distribution")
        original_df = create_histogram_data(original, 'Original')
        if not original_df.empty:
            st.bar_chart(
                data=original_df.set_index('Value')['Frequency'],
                color='#3498db',
                use_container_width=True
            )
            st.caption(f"Count: {len(original)} | Mean: {np.mean(original):.2f} | Std: {np.std(original):.2f}")
        else:
            st.info("No data to display")
    
    with col2:
        st.markdown("#### Synthetic Data Only")
        if len(synthetic) > 0:
            synthetic_df = create_histogram_data(synthetic, 'Synthetic')
            if not synthetic_df.empty:
                st.bar_chart(
                    data=synthetic_df.set_index('Value')['Frequency'],
                    color='#e74c3c',
                    use_container_width=True
                )
                st.caption(f"Count: {len(synthetic)} | Mean: {np.mean(synthetic):.2f} | Std: {np.std(synthetic):.2f}")
            else:
                st.info("No synthetic data to display")
        else:
            st.info("No synthetic data generated")
    
    with col3:
        st.markdown("#### Combined (Augmented)")
        augmented_df = create_histogram_data(augmented, 'Augmented')
        if not augmented_df.empty:
            st.bar_chart(
                data=augmented_df.set_index('Value')['Frequency'],
                color='#2ecc71',
                use_container_width=True
            )
            st.caption(f"Count: {len(augmented)} | Mean: {np.mean(augmented):.2f} | Std: {np.std(augmented):.2f}")
        else:
            st.info("No data to display")

def plot_numeric(data, title):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title)
    st.pyplot(fig)