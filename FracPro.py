import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import math

def main():
    st.set_page_config(
        page_title="Fracture Simulation App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Fracture Simulation Analysis")
    st.markdown("""
    This app simulates fracture propagation using KGD, PKN, and P3D models.
    Adjust the parameters in the sidebar to see how they affect fracture behavior.
    """)
    
    # Initialize session state for button clicks
    if 'run_2d_p3d' not in st.session_state:
        st.session_state.run_2d_p3d = False
    if 'run_p3d_growth' not in st.session_state:
        st.session_state.run_p3d_growth = False
    if 'run_pkn_growth' not in st.session_state:
        st.session_state.run_pkn_growth = False
    if 'run_kgd_growth' not in st.session_state:
        st.session_state.run_kgd_growth = False
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        
        # Parameter sliders with proper ranges
        G = st.slider(
            "Shear Modulus (G) [psi]", 
            min_value=1000000, 
            max_value=1000000000, 
            value=100000000, 
            step=1000000,
            help="Shear modulus of the rock formation"
        )
        
        v = st.slider(
            "Poisson's Ratio (v)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.25, 
            step=0.01,
            help="Poisson's ratio of the rock"
        )
        
        mu = st.slider(
            "Viscosity (mu) [cp]", 
            min_value=0.001, 
            max_value=1.0, 
            value=0.1, 
            step=0.001,
            help="Fluid viscosity"
        )
        
        Zi = st.slider(
            "In-situ Stress (Zi) [psi]", 
            min_value=1000, 
            max_value=10000, 
            value=5000, 
            step=100,
            help="In-situ stress condition"
        )
        
        Q = st.slider(
            "Flow Rate (Q) [bbl/min]", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Injection flow rate"
        )
        
        h = st.slider(
            "Height (h) [ft]", 
            min_value=10, 
            max_value=500, 
            value=100, 
            step=10,
            help="Fracture height"
        )
        
        rw = st.slider(
            "Wellbore Radius (rw) [ft]", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Wellbore radius"
        )
        
        tf = st.slider(
            "Simulation Time (tf) [min]", 
            min_value=1, 
            max_value=120, 
            value=60, 
            step=1,
            help="Total simulation time"
        )
        
        st.divider()
        
        # Action buttons
        st.header("📈 Visualization Options")
        if st.button("Run 2D & P3D Models", use_container_width=True):
            st.session_state.run_2d_p3d = True
            st.session_state.run_p3d_growth = False
            st.session_state.run_pkn_growth = False
            st.session_state.run_kgd_growth = False
            
        if st.button("Show P3D Growth", use_container_width=True):
            st.session_state.run_p3d_growth = True
            st.session_state.run_2d_p3d = False
            st.session_state.run_pkn_growth = False
            st.session_state.run_kgd_growth = False
            
        if st.button("Show PKN Growth", use_container_width=True):
            st.session_state.run_pkn_growth = True
            st.session_state.run_2d_p3d = False
            st.session_state.run_p3d_growth = False
            st.session_state.run_kgd_growth = False
            
        if st.button("Show KGD Growth", use_container_width=True):
            st.session_state.run_kgd_growth = True
            st.session_state.run_2d_p3d = False
            st.session_state.run_p3d_growth = False
            st.session_state.run_pkn_growth = False
    
    # Model functions with error handling
    def KGD_model(tf, G, Q, v, mu, Zi):
        """KGD model implementation"""
        try:
            t = np.linspace(0.001, tf, 100)
            # Avoid division by zero and negative values
            L = 0.48 * (8 * G * Q**3 / max((1 - v) * mu, 1e-10))**(1/6) * t**(2/3)
            W0 = 1.32 * (8 * (1 - v) * mu * Q**3 / max(G, 1e-10))**(1/6) * t**(1/3)
            # Ensure L is positive to avoid complex numbers
            L_positive = np.maximum(L, 1e-10)
            Pw = Zi + 0.96 * (2 * Q * mu * G**3 / max((1 - v)**3 * L_positive**2, 1e-10))**(1/4)
            return t, Pw, L, W0
        except Exception as e:
            st.error(f"Error in KGD model: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def PKN_model(tf, G, Q, v, mu, h):
        """PKN model implementation"""
        try:
            t = np.linspace(0.001, tf, 100)
            L = 0.68 * (G * Q**3 / max((1 - v) * mu * h**4, 1e-10))**(1/5) * t**(4/5)
            W0 = 2.5 * ((1 - v) * mu * Q**2 / max(G * h, 1e-10))**(1/5) * t**(1/5)
            Pw = 2.5 * (Q**2 * mu * G**4 / max((1 - v)**4 * h**6, 1e-10))**(1/5) * t**(1/5)
            return t, Pw, L, W0
        except Exception as e:
            st.error(f"Error in PKN model: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def P3D_model(tf, G, Q, mu, Zi, rw, h):
        """P3D model implementation"""
        try:
            t = np.linspace(0.001, tf, 100)
            R = 0.548 * (G * Q**3 / max(mu, 1e-10))**(1/9) * t**(4/9)
            W0 = 21 * (mu**2 * Q**3 / max(G**2, 1e-10))**(1/9) * t**(1/9)
            # Avoid log(0) and division by zero
            R_positive = np.maximum(R, rw + 1e-10)
            Pw = Zi - 5/(4 * np.pi) * (G * W0 / R_positive) * np.log(rw / R_positive)
            return t, Pw, R, W0
        except Exception as e:
            st.error(f"Error in P3D model: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Display current parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Shear Modulus (G)", f"{G/1e6:.1f} Mpsi")
        st.metric("Poisson's Ratio (v)", f"{v:.3f}")
    with col2:
        st.metric("Viscosity (mu)", f"{mu:.3f} cp")
        st.metric("In-situ Stress (Zi)", f"{Zi/1000:.1f} ksi")
    with col3:
        st.metric("Flow Rate (Q)", f"{Q:.1f} bbl/min")
        st.metric("Height (h)", f"{h:.0f} ft")
    with col4:
        st.metric("Wellbore Radius (rw)", f"{rw:.1f} ft")
        st.metric("Simulation Time (tf)", f"{tf:.0f} min")
    
    # Run simulations based on button clicks
    if st.session_state.run_2d_p3d:
        with st.spinner("Running 2D & P3D models..."):
            start_time = time.time()
            
            try:
                # Calculate all models
                t_kgd, p_kgd, L_kgd, W0_kgd = KGD_model(tf, G, Q, v, mu, Zi)
                t_pkn, p_pkn, L_pkn, W0_pkn = PKN_model(tf, G, Q, v, mu, h)
                t_p3d, p_p3d, R_p3d, W0_p3d = P3D_model(tf, G, Q, mu, Zi, rw, h)
                
                # Check if we have valid data
                if len(t_kgd) > 0 and len(t_pkn) > 0 and len(t_p3d) > 0:
                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            "KGD Model - Wellbore Pressure vs Time",
                            "PKN Model - Net Pressure vs Time", 
                            "P3D Model - Wellbore Pressure vs Time"
                        ),
                        vertical_spacing=0.1
                    )
                    
                    # Add traces
                    fig.add_trace(
                        go.Scatter(x=t_kgd, y=p_kgd, mode='lines', name='KGD', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=t_pkn, y=p_pkn, mode='lines', name='PKN', line=dict(color='red')),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=t_p3d, y=p_p3d, mode='lines', name='P3D', line=dict(color='green')),
                        row=3, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        title_text="WELLBORE PRESSURE vs TIME in 2D and P3D MODELS",
                        showlegend=True
                    )
                    
                    # Update axes labels
                    fig.update_xaxes(title_text="Time [min]", row=3, col=1)
                    fig.update_yaxes(title_text="Pressure [psi]", row=1, col=1)
                    fig.update_yaxes(title_text="Net Pressure [psi]", row=2, col=1)
                    fig.update_yaxes(title_text="Wellbore Pressure [psi]", row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show additional metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("KGD Final Pressure", f"{p_kgd[-1]:.1f} psi")
                        st.metric("KGD Final Length", f"{L_kgd[-1]:.1f} ft")
                    with col2:
                        st.metric("PKN Final Pressure", f"{p_pkn[-1]:.1f} psi")
                        st.metric("PKN Final Length", f"{L_pkn[-1]:.1f} ft")
                    with col3:
                        st.metric("P3D Final Pressure", f"{p_p3d[-1]:.1f} psi")
                        st.metric("P3D Final Radius", f"{R_p3d[-1]:.1f} ft")
                    
                    st.success(f"Simulation completed in {time.time() - start_time:.2f} seconds")
                else:
                    st.error("Error in model calculations. Please check your parameters.")
                    
            except Exception as e:
                st.error(f"Error during simulation: {e}")
    
    if st.session_state.run_p3d_growth:
        st.info("P3D Growth Visualization - This would show animated fracture growth over time")
        # Simple visualization for now
        try:
            t_p3d, p_p3d, R_p3d, W0_p3d = P3D_model(tf, G, Q, mu, Zi, rw, h)
            if len(t_p3d) > 0:
                fig_p3d = go.Figure()
                fig_p3d.add_trace(go.Scatter(x=t_p3d, y=R_p3d, mode='lines', name='Radius', line=dict(color='blue')))
                fig_p3d.add_trace(go.Scatter(x=t_p3d, y=W0_p3d, mode='lines', name='Width', line=dict(color='red')))
                fig_p3d.update_layout(title="P3D Model - Radius and Width vs Time", xaxis_title="Time [min]", yaxis_title="Value")
                st.plotly_chart(fig_p3d, use_container_width=True)
        except Exception as e:
            st.error(f"Error in P3D growth visualization: {e}")
        
    if st.session_state.run_pkn_growth:
        st.info("PKN Growth Visualization - This would show animated fracture growth over time")
        try:
            t_pkn, p_pkn, L_pkn, W0_pkn = PKN_model(tf, G, Q, v, mu, h)
            if len(t_pkn) > 0:
                fig_pkn = go.Figure()
                fig_pkn.add_trace(go.Scatter(x=t_pkn, y=L_pkn, mode='lines', name='Length', line=dict(color='blue')))
                fig_pkn.add_trace(go.Scatter(x=t_pkn, y=W0_pkn, mode='lines', name='Width', line=dict(color='red')))
                fig_pkn.update_layout(title="PKN Model - Length and Width vs Time", xaxis_title="Time [min]", yaxis_title="Value")
                st.plotly_chart(fig_pkn, use_container_width=True)
        except Exception as e:
            st.error(f"Error in PKN growth visualization: {e}")
        
    if st.session_state.run_kgd_growth:
        st.info("KGD Growth Visualization - This would show animated fracture growth over time")
        try:
            t_kgd, p_kgd, L_kgd, W0_kgd = KGD_model(tf, G, Q, v, mu, Zi)
            if len(t_kgd) > 0:
                fig_kgd = go.Figure()
                fig_kgd.add_trace(go.Scatter(x=t_kgd, y=L_kgd, mode='lines', name='Length', line=dict(color='blue')))
                fig_kgd.add_trace(go.Scatter(x=t_kgd, y=W0_kgd, mode='lines', name='Width', line=dict(color='red')))
                fig_kgd.update_layout(title="KGD Model - Length and Width vs Time", xaxis_title="Time [min]", yaxis_title="Value")
                st.plotly_chart(fig_kgd, use_container_width=True)
        except Exception as e:
            st.error(f"Error in KGD growth visualization: {e}")
    
    # Always show model comparison
    st.subheader("📊 Model Comparison")
    
    try:
        t_kgd, p_kgd, L_kgd, W0_kgd = KGD_model(tf, G, Q, v, mu, Zi)
        t_pkn, p_pkn, L_pkn, W0_pkn = PKN_model(tf, G, Q, v, mu, h)
        t_p3d, p_p3d, R_p3d, W0_p3d = P3D_model(tf, G, Q, mu, Zi, rw, h)
        
        if len(t_kgd) > 0 and len(t_pkn) > 0 and len(t_p3d) > 0:
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Scatter(
                x=t_kgd, y=p_kgd, mode='lines', name='KGD Model',
                line=dict(color='blue', width=2)
            ))
            
            fig_comp.add_trace(go.Scatter(
                x=t_pkn, y=p_pkn, mode='lines', name='PKN Model',
                line=dict(color='red', width=2)
            ))
            
            fig_comp.add_trace(go.Scatter(
                x=t_p3d, y=p_p3d, mode='lines', name='P3D Model',
                line=dict(color='green', width=2)
            ))
            
            fig_comp.update_layout(
                title="Pressure Comparison Across Models",
                xaxis_title="Time [min]",
                yaxis_title="Pressure [psi]",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Model comparison data not available. Please check your parameters.")
            
    except Exception as e:
        st.error(f"Error in model comparison: {e}")
    
    # Additional information
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        ### Fracture Propagation Models
        
        **KGD Model (Khristianovic-Geertsma-de Klerk):**
        - Assumes constant height fracture
        - Suitable for short, wide fractures
        - Named after the developers
        
        **PKN Model (Perkins-Kern-Nordgren):**
        - Assumes elliptical cross-section
        - Good for long, narrow fractures
        - Commonly used in industry
        
        **P3D Model (Pseudo-3D):**
        - Accounts for height growth
        - More realistic for complex formations
        - Computationally intensive
        
        These models help predict fracture geometry and pressure distribution
        during hydraulic fracturing operations.
        """)
    
    # Footer
    st.divider()
    st.caption("""
    🛠️ Built with Streamlit | 📊 Visualization with Plotly | 
    ⚗️ Fracture mechanics simulation based on classical models
    """)

if __name__ == "__main__":
    main()
