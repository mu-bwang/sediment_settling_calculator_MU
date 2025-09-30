# Import necessary modules
from flask import Flask, render_template, request, url_for
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from sklearn.impute import SimpleImputer
import sediment_type
import ref_water

# Initialize Flask app
app = Flask(__name__)

# Load scaler and RF model for single particle
try:
    single_scaler = joblib.load('Particles_single_checkpoint/Particles_6features_scaler.pkl')
    single_rf_model = joblib.load('Particles_single_checkpoint/Particles_6features_rf_model.pkl')
except FileNotFoundError as e:
    print(f"Error loading single particle models: {e}")
    print("Please ensure the following files exist:")
    print("- Particles_single_checkpoint/Particles_6features_scaler.pkl")
    print("- Particles_single_checkpoint/Particles_6features_rf_model.pkl")
    exit(1)

# Load scaler and RF model for multiple particles
try:
    multiple_scaler = joblib.load('Particles_single_checkpoint_1600/Particles_6features_scaler.pkl')
    multiple_rf_model = joblib.load('Particles_single_checkpoint_1600/Particles_6features_rf_model.pkl')
except FileNotFoundError as e:
    print(f"Error loading multiple particle models: {e}")
    print("Please ensure the following files exist:")
    print("- Particles_single_checkpoint_1600/Particles_6features_scaler.pkl")
    print("- Particles_single_checkpoint_1600/Particles_6features_rf_model.pkl")
    exit(1)

# Constants
m = 6.0  # power-law coefficient
kappa = 0.41  # von Karman constant

# Set font settings
FS = 16
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = FS
plt.rcParams['axes.labelsize'] = FS
plt.rcParams['axes.titlesize'] = FS
plt.rcParams['legend.fontsize'] = FS
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS
plt.rcParams['figure.dpi'] = 500

def calculate_sediment_properties(velocity, water_depth, diameter):
    Um = (m + 1) / m * velocity
    shear_velocity = Um * kappa / m
    water = ref_water.water(20)
    sand_particle = sediment_type.sand(water, diameter)
    settling_velocity = np.abs(sand_particle.ws[0])
    Ro_beta = min(3, 1 + 2 * (settling_velocity / shear_velocity) ** 2)
    Ro = settling_velocity / (Ro_beta * kappa * shear_velocity)

    if Ro < 2.5:
        return shear_velocity, settling_velocity, Ro, False
    return shear_velocity, settling_velocity, Ro, True

def safe_lognorm_calculation(median_rf, variance_rf):
    """
    Safely calculate lognormal distribution parameters with error checking.
    Returns None if calculations would result in mathematical errors.
    """
    try:
        # Check for valid inputs
        if median_rf <= 0:
            return None
        if variance_rf <= 0:
            return None
        
        # Calculate mu (mean of log distribution)
        miu = np.log(median_rf)
        
        # Calculate scale and shape for lognormal distribution
        scale_rf = median_rf  # Since median = exp(mean_log)
        
        # Check for division by zero
        if median_rf ** 2 == 0:
            return None
            
        # Check if the argument to log is valid
        log_arg = 1 + variance_rf / (median_rf ** 2)
        if log_arg <= 0:
            return None
            
        shape_rf = np.sqrt(np.log(log_arg))  # Correct shape (std dev in log space)
        
        return miu, scale_rf, shape_rf
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    calculated_values = None
    plot_filename = None
    velocity, water_depth, diameter = None, None, None
    mode = "single"
    suspension_message = None

    if request.method == "POST":
        mode = request.form['mode']
        velocity = request.form.get('feature1')
        water_depth = request.form.get('feature2')
        diameter = request.form.get('feature3', "")

        try:
            velocity = float(velocity) if velocity else None
            water_depth = float(water_depth) if water_depth else None
            diameter = float(diameter) if diameter else None
        except ValueError:
            return "Please enter valid numeric values."

        # Validate common parameters
        if (velocity is None or velocity < 0.1 or velocity > 1.0 or
            water_depth is None or water_depth < 0.5 or water_depth > 10.0):
            suspension_message = "Velocity or water depth parameters outside the range! No output plot."
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
            plt.axis('off')
            plot_filename = 'parameter_error_plot.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()
            return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                   calculated_values=None, mode=mode, velocity=velocity,
                                   water_depth=water_depth, diameter=diameter,
                                   suspension_message=suspension_message)

        # Validate diameter for single mode only
        if mode == "single" and (diameter is None or diameter < 0.00001 or diameter > 0.02):
            suspension_message = "Diameter parameter outside the range! No output plot."
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
            plt.axis('off')
            plot_filename = 'parameter_error_plot.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()
            return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                   calculated_values=None, mode=mode, velocity=velocity,
                                   water_depth=water_depth, diameter=diameter,
                                   suspension_message=suspension_message)

        if mode == "single" and velocity and water_depth and diameter:
            shear_velocity, settling_velocity, Ro, valid = calculate_sediment_properties(velocity, water_depth,
                                                                                         diameter)
            if not valid:
                suspension_message = "particles are suspended! No output plot."
                # Create a simple plot to indicate suspension
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
                plt.axis('off')
                plot_filename = 'suspension_plot.png'
                plt.savefig(f'static/{plot_filename}')
                plt.close()
                return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                       calculated_values=None, mode=mode, velocity=velocity,
                                       water_depth=water_depth, diameter=diameter,
                                       suspension_message=suspension_message)

            calculated_values = {'shear_velocity': shear_velocity, 'settling_velocity': settling_velocity, 'Ro': Ro}
            predict_features = np.array(
                [velocity, water_depth, diameter, shear_velocity, settling_velocity, Ro]).reshape(1, -1)
            predict_features_scaled = single_scaler.transform(predict_features)
            predict_features_imputed = SimpleImputer(strategy='mean').fit_transform(predict_features_scaled)
            rf_predictions = single_rf_model.predict(predict_features_imputed)

            # Plot
            # Assuming rf_predictions[:, 0] = median, rf_predictions[:, 1] = variance
            median_rf = rf_predictions[:, 0]
            variance_rf = rf_predictions[:, 1]
            
            # Safely calculate lognormal parameters
            lognorm_params = safe_lognorm_calculation(median_rf[0], variance_rf[0])
            if lognorm_params is None:
                suspension_message = "Mathematical error in calculations! Unable to generate plot."
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
                plt.axis('off')
                plot_filename = 'calculation_error_plot.png'
                plt.savefig(f'static/{plot_filename}')
                plt.close()
                return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                       calculated_values=calculated_values, mode=mode, velocity=velocity,
                                       water_depth=water_depth, diameter=diameter,
                                       suspension_message=suspension_message)
            
            miu, scale_rf, shape_rf = lognorm_params

            x = np.linspace(
                lognorm.ppf(0.0001, s=shape_rf, scale=scale_rf),  # 1st percentile
                lognorm.ppf(0.9999, s=shape_rf, scale=scale_rf),  # 90th percentile
                1000  # Number of points to generate
            )

            # Calculate PDF
            pdf = lognorm.pdf(x, s=shape_rf, scale=scale_rf)

            plt.figure(figsize=(8, 6))
            plt.plot(x, pdf, 'g--', lw=2, label='lognorm_RF')
            plt.text(
                0.9, 0.75,
                f'$\\mu$ (RF): {miu:.3f}\n$\\sigma$ (RF): {shape_rf:.3f}\n'
                f'Median (RF): {rf_predictions[0, 0]:.3f} m\nVariance (RF): {rf_predictions[0, 1]:.3f} m²',
                transform=plt.gca().transAxes,
                fontsize=FS,
                verticalalignment='top',
                horizontalalignment='right'
            )

            plt.xlabel("Distance (m)")
            plt.ylabel("PDF")
            plt.legend()
            plot_filename = 'plot_single.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()

        elif mode == "multiple" and velocity and water_depth:
            predict_features = np.array([[velocity, water_depth]])
            predict_features_scaled = multiple_scaler.transform(predict_features)
            predict_features_imputed = SimpleImputer(strategy='mean').fit_transform(predict_features_scaled)
            rf_predictions = multiple_rf_model.predict(predict_features_imputed)

            # Assuming rf_predictions[:, 0] = median, rf_predictions[:, 1] = variance
            median_rf = rf_predictions[:, 0]
            variance_rf = rf_predictions[:, 1]
            
            # Safely calculate lognormal parameters
            lognorm_params = safe_lognorm_calculation(median_rf[0], variance_rf[0])
            if lognorm_params is None:
                suspension_message = "Mathematical error in calculations! Unable to generate plot."
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, suspension_message, fontsize=FS, ha='center', va='center')
                plt.axis('off')
                plot_filename = 'calculation_error_plot.png'
                plt.savefig(f'static/{plot_filename}')
                plt.close()
                return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                                       calculated_values=None, mode=mode, velocity=velocity,
                                       water_depth=water_depth, diameter=diameter,
                                       suspension_message=suspension_message)
            
            miu, scale_rf, shape_rf = lognorm_params

            x = np.linspace(
                lognorm.ppf(0.0001, s=shape_rf, scale=scale_rf),  # 1st percentile
                lognorm.ppf(0.9999, s=shape_rf, scale=scale_rf),  # 90th percentile
                1000  # Number of points to generate
            )

            # Calculate PDF
            pdf = lognorm.pdf(x, s=shape_rf, scale=scale_rf)
            plt.figure(figsize=(8, 6))
            plt.plot(x, pdf, 'g--', lw=2, label='lognorm_RF')
            plt.text(
                0.9, 0.75,
                f'$\\mu$ (RF): {miu:.3f}\n$\\sigma$ (RF): {shape_rf:.3f}\n'
                f'Median (RF): {rf_predictions[0, 0]:.3f} m\nVariance (RF): {rf_predictions[0, 1]:.3f} m²',
                transform=plt.gca().transAxes,
                fontsize=FS,
                verticalalignment='top',
                horizontalalignment='right'
            )

            plt.xlabel("Distance (m)")
            plt.ylabel("PDF")
            plt.legend()
            plot_filename = 'plot_multiple.png'
            plt.savefig(f'static/{plot_filename}')
            plt.close()

        return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                               calculated_values=calculated_values, mode=mode, velocity=velocity,
                               water_depth=water_depth, diameter=diameter, suspension_message=suspension_message)

    return render_template("index.html", plot_url=None, mode=mode, velocity=None, water_depth=None, diameter=None,
                           suspension_message=None)

if __name__ == "__main__":
    app.run(debug=True)
