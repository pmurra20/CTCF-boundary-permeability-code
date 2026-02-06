import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
import seaborn as sns
from scipy.stats import chisquare
import sys
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib as mpl
import joblib
from sklearn.model_selection import KFold


def bin_size(data, binwidth):
    return np.arange(min(data), max(data) + binwidth, binwidth)

def aic_bic_hmm_states(df, max_no_states):
    # Reshape force data for HMM input
    X = df.values.reshape(-1, 1)

    # Lists to store AIC and BIC values for different number of states
    aic_values = []
    bic_values = []
    num_states = range(1, max_no_states)  # Try 1 to max_no_states - 1

    # Use tqdm for progress tracking
    for n in tqdm(num_states, desc="Calculating AIC/BIC for each state"):
        # Initialize the HMM
        model = GaussianHMM(n_components=n, covariance_type="full", n_iter=100)
        
        # Fit the model
        model.fit(X)
        
        # Calculate AIC and BIC
        log_likelihood = model.score(X)  # Log likelihood of the fitted model
        num_params = n**2 + 2 * n * X.shape[1] - 1  # Estimated number of parameters in the model
        
        aic = 2 * num_params - 2 * log_likelihood
        bic = np.log(X.shape[0]) * num_params - 2 * log_likelihood
        
        aic_values.append(aic)
        bic_values.append(bic)

    # Plot AIC and BIC values
    plt.figure(figsize=(5, 3))
    plt.plot(num_states, aic_values, label='AIC', marker='o', linestyle='--', color='dimgrey')
    plt.plot(num_states, bic_values, label='BIC', marker='o', linestyle='--', color='firebrick')
    plt.xlabel('Number of States')
    plt.ylabel('Criterion Value')
    plt.title('AIC and BIC for Model Selection')
    plt.legend(frameon=False)
    plt.show()

    return aic_values, bic_values


def aic_bic_hmm_states_combined(traces, max_no_states, cross_val=False, n_splits=5):
    # Combine force data from all traces into a single sequence without normalization
    combined_force = np.concatenate([trace['force_corrected (pN)'].values for trace in traces])
    
    # Remove any NaN values from the combined force data
    valid_indices = ~np.isnan(combined_force)
    combined_force = combined_force[valid_indices]

    # Reshape force data for HMM input (HMM expects 2D input)
    X = combined_force.reshape(-1, 1)

    # Lists to store AIC, BIC, and log-likelihood values for different numbers of states
    aic_values = []
    bic_values = []
    log_likelihood_values = []
    num_states = range(1, max_no_states)  # Try 1 to max_no_states - 1

    # Use tqdm for progress tracking
    for n in tqdm(num_states, desc="Calculating AIC/BIC for each state"):
        model = GaussianHMM(n_components=n, covariance_type="full", n_iter=200)
        
        if cross_val:
            # Cross-validation
            kf = KFold(n_splits=n_splits)
            log_likelihood_cv = []

            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                model.fit(X_train)
                log_likelihood_cv.append(model.score(X_val))

            log_likelihood = np.mean(log_likelihood_cv)
        else:
            model.fit(X)
            log_likelihood = model.score(X)  # Log likelihood of the fitted model

        # Calculate AIC and BIC
        num_params = n**2 + 2 * n * X.shape[1] - 1  # Estimated number of parameters in the model
        aic = 2 * num_params - 2 * log_likelihood
        bic = np.log(X.shape[0]) * num_params - 2 * log_likelihood
        
        aic_values.append(aic)
        bic_values.append(bic)
        log_likelihood_values.append(log_likelihood)

    # Plot AIC and BIC values
    plt.figure(figsize=(5, 3))
    plt.plot(num_states, aic_values, label='AIC', marker='o', linestyle='--', color='dimgrey')
    plt.plot(num_states, bic_values, label='BIC', marker='o', linestyle='--', color='firebrick')
    plt.xlabel('Number of States')
    plt.ylabel('Criterion Value')
    plt.title('AIC and BIC for Model Selection')
    plt.legend(frameon=False)
    plt.show()

    # Plot Log-Likelihood
    plt.figure(figsize=(5, 3))
    plt.plot(num_states, log_likelihood_values, label='Log-Likelihood', marker='o', linestyle='--', color='darkgreen')
    plt.xlabel('Number of States')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood for Model Selection')
    plt.legend(frameon=False)
    plt.show()

    return aic_values, bic_values, log_likelihood_values

def downsample(df, downsampling):
    df['time_delta'] = pd.to_timedelta(df['time (seconds)'], unit='s')
    df.set_index('time_delta', inplace=True)

    # Downsample data to 500,000 microseconds (0.5 seconds)
    downsampled_df = df.resample(downsampling).mean()  # '500000U' represents 500,000 microseconds (0.5 seconds)

    # Optionally, drop any rows with NaN values that might have been introduced
    downsampled_df.dropna(inplace=True)

    return downsampled_df

def linear_correction(df, intercept, window_size=5000):
    # Assuming your DataFrame is already loaded as df (df['time (seconds)'], df['force (pN)'])
    time = df['time (seconds)'].values
    force = df['force (pN)'].values

    # Step 1: Apply a rolling window to smooth the force data
    force_smooth = pd.Series(force).rolling(window=window_size, min_periods=1, center=True).mean().values

    # Step 2: Fit a linear line (linear regression) to the smoothed data
    slope, intercept = np.polyfit(time, force_smooth, 1)

    # Step 3: Calculate the linear drift component using the fitted line from smoothed data
    fitted_line = slope * time + intercept

    # Step 4: Correct the original force data (not the smoothed one) while keeping the intercept at 12
    initial_force_corrected = force[0]  # Starting force value of the original data
    force_corrected = force - (fitted_line - initial_force_corrected)

    # Step 5: Adjust the corrected force data so that its fitted line has an intercept at exactly 12
    corrected_slope, corrected_intercept = np.polyfit(time, force_corrected, 1)
    intercept_adjustment = intercept - corrected_intercept
    force_corrected += intercept_adjustment

    # Step 6: Refit the corrected data to check the new intercept
    final_corrected_slope, final_corrected_intercept = np.polyfit(time, force_corrected, 1)
    corrected_fitted_line = final_corrected_slope * time + final_corrected_intercept

    # Step 7: Plot the original force data, smoothed data, the fitted linear trend, and the corrected force data
    fig, ax = plt.subplots(3, 1, figsize=(7, 7))

    # Plot the original force data
    ax[0].plot(time, force, label="Original Force", color='black', alpha=0.5)

    # Plot the smoothed force data
    ax[0].plot(time, force_smooth, label=f"Smoothed Force (window={window_size})", color='green', lw=1.5)

    # Plot the fitted linear trend
    ax[1].plot(time, force, label="Original Force", color='black', alpha=0.5)
    ax[1].plot(time, fitted_line, label=f"Fitted Line (y = {slope:.3f}x + {intercept:.2f})", color='red', lw=2)

    # Plot the corrected force data with its final fit
    ax[2].plot(time, force_corrected, label="Corrected Force (Intercept at 12.00)", color='blue')
    ax[2].plot(time, corrected_fitted_line, label=f"Fit to Corrected Data (y = {final_corrected_slope:.3f}x + {final_corrected_intercept:.2f})", color='orange', lw=2, linestyle='--')

    # Customize the plots
    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Force (pN)')
    ax[0].legend()

    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_ylabel('Force (pN)')
    ax[1].legend()

    ax[2].set_xlabel('Time (seconds)')
    ax[2].set_ylabel('Force (pN)')
    ax[2].legend()

    plt.suptitle('Force Over Time with Linear Drift Correction Using Smoothed Data')
    plt.tight_layout()
    plt.show()

    # Optionally: Add the corrected force data back into the DataFrame
    df['force_corrected (pN)'] = force_corrected

    return df

def linear_correction_bp(df, window_size=5000):
    # Assuming your DataFrame is already loaded as df (df['time (seconds)'], df['force (pN)'])
    time = df['time'].values
    force = df['bp'].values

    # Step 1: Apply a rolling window to smooth the force data
    force_smooth = pd.Series(force).rolling(window=window_size, min_periods=1, center=True).mean().values

    # Step 2: Fit a linear line (linear regression) to the smoothed data
    slope, intercept = np.polyfit(time, force_smooth, 1)

    # Step 3: Calculate the linear drift component using the fitted line from smoothed data
    fitted_line = slope * time + intercept

    # Step 4: Correct the original force data (not the smoothed one) while keeping the intercept at 12
    initial_force_corrected = force[0]  # Starting force value of the original data
    force_corrected = force - (fitted_line - initial_force_corrected)

    # Step 5: Adjust the corrected force data so that its fitted line has an intercept at exactly 12
    corrected_slope, corrected_intercept = np.polyfit(time, force_corrected, 1)
    intercept_adjustment = 0 - corrected_intercept
    force_corrected += intercept_adjustment

    # Step 6: Refit the corrected data to check the new intercept
    final_corrected_slope, final_corrected_intercept = np.polyfit(time, force_corrected, 1)
    corrected_fitted_line = final_corrected_slope * time + final_corrected_intercept

    # Step 7: Plot the original force data, smoothed data, the fitted linear trend, and the corrected force data
    fig, ax = plt.subplots(3, 1, figsize=(7, 7))

    # Plot the original force data
    ax[0].plot(time, force, label="Original bp", color='black', alpha=0.5)

    # Plot the smoothed force data
    ax[0].plot(time, force_smooth, label=f"Smoothed bp (window={window_size})", color='green', lw=1.5)

    # Plot the fitted linear trend
    ax[1].plot(time, force, label="Original bp", color='black', alpha=0.5)
    ax[1].plot(time, fitted_line, label=f"Fitted Line (y = {slope:.3f}x + {intercept:.2f})", color='red', lw=2)

    # Plot the corrected force data with its final fit
    ax[2].plot(time, force_corrected, label="Corrected bp (Intercept at 12.00)", color='blue')
    ax[2].plot(time, corrected_fitted_line, label=f"Fit to Corrected Data (y = {final_corrected_slope:.3f}x + {final_corrected_intercept:.2f})", color='orange', lw=2, linestyle='--')

    # Customize the plots
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('bp')
    ax[0].legend()

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('bp')
    ax[1].legend()

    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('bp')
    ax[2].legend()

    plt.suptitle('Bp Over Time with Linear Drift Correction Using Smoothed Data')
    plt.tight_layout()
    plt.show()

    # Optionally: Add the corrected force data back into the DataFrame
    df['bp_corrected'] = force_corrected

    return df

def find_hmm_states(df, no_states, experiment, iterations=1000):
    # Load your DataFrame (assuming df_after is already loaded)
    time = df['time (seconds)'].values
    force = df['force_corrected (pN)'].values

    # Reshape force data for HMM input (HMM expects 2D input)
    X = force.reshape(-1, 1)

    # Initialize and train HMM
    n_components = no_states  # Number of hidden states (adjust as needed)
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=iterations)

    # Fit the model on the data
    model.fit(X)

    # Predict the hidden states
    hidden_states = model.predict(X)

    # Calculate the mean force for each hidden state
    state_means = np.array([X[hidden_states == i].mean() for i in range(n_components)])

    # Print the mean force corresponding to each hidden state
    for i, mean_force in enumerate(state_means):
        print(f"State {i+1}: Mean Force = {mean_force:.2f} pN")

    # Create a step function for the hidden states
    step_function = np.zeros_like(force)
    for i in range(n_components):
        step_function[hidden_states == i] = state_means[i]

    # Plot the original force data and overlay the step function
    plt.figure(figsize=(5.5, 2.5))

    # Plot the original force data
    plt.plot(time, force, label="Force (pN)", color='black')

    # Overlay the step function (representing the states)
    plt.step(time, step_function, where='mid', label='HMM State Force Levels', color='red', lw=1)

    # Customize plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Force (pN)')
    plt.xlim(40,40.2)
    plt.title(f'HMM Segmentation of Force Over Time ({experiment})')
    plt.legend()
    plt.show()

    # 1. Calculate the abundance of each state
    unique, counts = np.unique(hidden_states, return_counts=True)
    state_abundances = counts / len(hidden_states)  # Abundance is the fraction of total time spent in each state

    # 2. Exclude self-transitions and calculate new transition probabilities
    transition_counts = np.zeros((n_components, n_components))

    # Loop through hidden states and count transitions between different states
    for i in range(1, len(hidden_states)):
        prev_state = hidden_states[i-1]
        curr_state = hidden_states[i]
        if prev_state != curr_state:  # Only count transitions between different states
            transition_counts[prev_state, curr_state] += 1

    # Calculate row-wise transition probabilities excluding same-state transitions
    transition_matrix_no_self = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    # Add both transition counts and percentages to the heatmap
    labels = np.zeros_like(transition_matrix_no_self, dtype=object)
    for i in range(n_components):
        for j in range(n_components):
            if transition_counts[i, j] > 0:
                labels[i, j] = f'{transition_matrix_no_self[i, j]:.2%}\n(N = {int(transition_counts[i, j])})'

    # 3. Calculate the average and full distribution of lifetimes for each state
    lifetimes = []
    all_lifetimes = {state: [] for state in range(n_components)}  # Store all lifetimes for each state

    for state in range(n_components):
        state_durations = []
        in_state = False
        start_time = 0

        for i in range(1, len(hidden_states)):
            if hidden_states[i] == state and not in_state:
                # Start of the state
                start_time = time[i]
                in_state = True
            elif hidden_states[i] != state and in_state:
                # End of the state
                duration = time[i] - start_time
                state_durations.append(duration)
                all_lifetimes[state].append(duration)
                in_state = False

        if state_durations:
            lifetimes.append(np.mean(state_durations))
        else:
            lifetimes.append(0)

    # 4. Calculate and print the mean force for each state
    mean_forces = [np.mean(force[hidden_states == i]) for i in range(n_components)]
        
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.5))
    ax[0].bar(range(1, n_components+1), mean_forces, color='dimgrey')
    ax[0].set_xlabel('State')
    ax[0].set_ylabel('Force (pN)')
    ax[0].set_title('Average Force')
    ax[0].set_xticks(range(1, n_components+1))

    ax[1].bar(range(1, n_components+1), state_abundances, color='skyblue')
    ax[1].set_xlabel('State')
    ax[1].set_ylabel('Frequency (%)')
    ax[1].set_title('Frequency')
    ax[1].set_xticks(range(1, n_components+1))

    # Plot the average lifetime of each state
    ax[2].bar(range(1, n_components+1), lifetimes, color='salmon')
    ax[2].set_xlabel('State')
    ax[2].set_ylabel('Average Lifetime (s)')
    ax[2].set_title('Average Lifetime')
    ax[2].set_xticks(range(1, n_components+1))
    plt.suptitle(experiment)
    plt.tight_layout()
    plt.show()

    # 5. Plot distribution of lifetimes for each state in different ranges
    fig, axs = plt.subplots(n_components, 2, figsize=(6, 3 * n_components))

    for i in range(n_components):
        # First range (e.g., between 0 and 0.125 seconds)
        axs[i, 0].hist(all_lifetimes[i], bins=30, range=(0, 0.125), color='firebrick', edgecolor='black', alpha=0.7)
        axs[i, 0].set_xlim(-0.005, 0.125)
        axs[i, 0].set_title(f'Lifetimes State {i+1} (0-0.125s)')
        axs[i, 0].set_ylabel('Frequency')
        axs[i, 0].set_xlabel('Lifetime (seconds)')

        # Second range (e.g., between 1 and 10 seconds)
        axs[i, 1].hist(all_lifetimes[i], bins=20, range=(0.5, 20), color='darkgrey', edgecolor='black', alpha=0.7)
        axs[i, 1].set_xlim(0.5, 20)
        axs[i, 1].set_title(f'Lifetimes State {i+1} (1-20s)')
        axs[i, 1].set_ylabel('Frequency')
        axs[i, 1].set_xlabel('Lifetime (seconds)')

    plt.suptitle(experiment)
    plt.tight_layout()
    plt.show()

    return all_lifetimes

def find_hmm_states_combined(traces, no_states, experiment, iterations=1000):
    if len(traces) != 2:
        raise ValueError("This function expects exactly two traces.")

    # Normalize each trace by subtracting the average force in the initial period
    normalized_traces = []
    initial_period = 5  # Define the initial period in seconds to calculate the average

    for trace in traces:
        initial_forces = trace[trace['time (seconds)'] <= initial_period]['force_corrected (pN)'].values
        if len(initial_forces) > 0 and not np.isnan(initial_forces).all():
            initial_force_avg = np.nanmean(initial_forces)
        else:
            initial_force_avg = 0
        
        trace['normalized_force'] = trace['force_corrected (pN)'] - initial_force_avg
        normalized_traces.append(trace)

    # Combine normalized force data
    combined_force = np.concatenate([trace['normalized_force'].values for trace in normalized_traces])
    if np.isnan(combined_force).any():
        raise ValueError("Normalization resulted in NaN values.")

    # Create continuous time sequence
    combined_time = []
    current_end_time = 0
    for trace in normalized_traces:
        time = trace['time (seconds)'].values
        time_step = np.median(np.diff(time))
        adjusted_time = np.arange(len(time)) * time_step + current_end_time
        combined_time.append(adjusted_time)
        current_end_time = adjusted_time[-1] + time_step
    combined_time = np.concatenate(combined_time)

    # Reshape force data for HMM input
    X = combined_force.reshape(-1, 1)

    # Initialize and train HMM
    n_components = no_states
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=iterations)
    model.fit(X)
    hidden_states = model.predict(X)

    # Calculate mean force for each hidden state
    state_means = np.array([X[hidden_states == i].mean() for i in range(n_components)])

    # Sort states in order of decreasing mean force
    sorted_indices = np.argsort(-state_means)
    sorted_hidden_states = np.zeros_like(hidden_states)
    for new_label, old_label in enumerate(sorted_indices):
        sorted_hidden_states[hidden_states == old_label] = new_label
    
    state_means_sorted = state_means[sorted_indices]

    # Calculate transition matrix excluding self-transitions
    transition_counts = np.zeros((n_components, n_components))
    for i in range(1, len(sorted_hidden_states)):
        prev_state = sorted_hidden_states[i-1]
        curr_state = sorted_hidden_states[i]
        if prev_state != curr_state:
            transition_counts[prev_state, curr_state] += 1
    transition_matrix_no_self = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    # Plot transition matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(transition_matrix_no_self, annot=True, cmap='Blues', fmt='.2f', cbar=True,
                xticklabels=[f'State {i+1}' for i in range(n_components)],
                yticklabels=[f'State {i+1}' for i in range(n_components)])
    plt.title(f'Transition Probability Matrix Excluding Self-Transitions ({experiment})')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.show()

    # Plot original force data and HMM states
    plt.figure(figsize=(10, 5))
    first_trace_length = len(normalized_traces[0])
    end_first_trace_time = combined_time[first_trace_length - 1]
    step_function = np.zeros_like(combined_force)
    for i in range(n_components):
        step_function[sorted_hidden_states == i] = state_means_sorted[i]

    plt.plot(combined_time, combined_force, label="Combined Force (pN)", alpha=0.7, color='blue')
    plt.step(combined_time, step_function, where='mid', label="HMM State Levels", lw=1.5, color='red')
    plt.axvline(x=end_first_trace_time, color='black', linestyle='--', label="Trace Separation")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Force (pN)')
    plt.title(f'HMM Segmentation of Combined Force Over Time ({experiment})')
    plt.legend()
    plt.show()

    # Plot force distributions for each state
    fig, axs = plt.subplots(n_components, 2, figsize=(12, n_components * 3))
    for i in range(n_components):
        # Plot distribution of forces for each state
        axs[i, 0].hist(combined_force[sorted_hidden_states == i], bins=30, color='blue', alpha=0.7)
        axs[i, 0].set_title(f'Force Distribution for State {i+1}')
        axs[i, 0].set_xlabel('Force (pN)')
        axs[i, 0].set_ylabel('Frequency')

        # Calculate lifetimes for each state
        state_lifetimes = []
        in_state = False
        start_time = 0
        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                state_lifetimes.append(combined_time[j] - start_time)
                in_state = False

        # Plot distribution of lifetimes for each state
        axs[i, 1].hist(state_lifetimes, bins=30, color='grey', alpha=0.7)
        axs[i, 1].set_title(f'Lifetime Distribution for State {i+1}')
        axs[i, 1].set_xlabel('Lifetime (seconds)')
        axs[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Plot bar plots with error bars and annotations
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    mean_forces, std_forces = [], []
    state_abundances, state_abundances_std = [], []
    lifetimes = []

    for state in range(n_components):
        state_forces = combined_force[sorted_hidden_states == state]
        mean_forces.append(state_forces.mean())
        std_forces.append(state_forces.std())

        state_abundances.append(len(state_forces) / len(combined_force))

        # Calculate lifetimes
        state_lifetimes = []
        in_state = False
        start_time = 0
        for i in range(len(sorted_hidden_states)):
            if sorted_hidden_states[i] == state and not in_state:
                start_time = combined_time[i]
                in_state = True
            elif sorted_hidden_states[i] != state and in_state:
                state_lifetimes.append(combined_time[i] - start_time)
                in_state = False
        
        if state_lifetimes:
            lifetimes.append(np.mean(state_lifetimes))
        else:
            lifetimes.append(0)

    # Plot average force with error bars
    ax[0].bar(range(1, n_components + 1), mean_forces, yerr=std_forces, color='dimgrey', capsize=5)
    ax[0].set_xlabel('State')
    ax[0].set_ylabel('Force (pN)')
    ax[0].set_title('Average Force')
    ax[0].set_xticks(range(1, n_components + 1))

    # Plot state abundances
    ax[1].bar(range(1, n_components + 1), state_abundances, color='skyblue', capsize=5)
    ax[1].set_xlabel('State')
    ax[1].set_ylabel('Frequency (%)')
    ax[1].set_title('Frequency')
    ax[1].set_xticks(range(1, n_components + 1))

    # Plot average lifetime without error bars, annotate values
    ax[2].bar(range(1, n_components + 1), lifetimes, color='salmon')
    ax[2].set_xlabel('State')
    ax[2].set_ylabel('Average Lifetime (s)')
    ax[2].set_title('Average Lifetime')
    ax[2].set_xticks(range(1, n_components + 1))
    
    # Annotate bars with average lifetime values
    for i, lifetime in enumerate(lifetimes):
        ax[2].text(i + 1, lifetime, f'{lifetime:.2f}', ha='center', va='bottom')

    plt.suptitle(experiment)
    plt.tight_layout()
    plt.show()

    return lifetimes

def HMM_fit_force(traces, random_state, no_states, no_iterations, savepath):
    np.random.seed(random_state)
    all_force_data = np.concatenate([trace['force_corrected (pN)'].values for trace in traces])

    all_time_data = []
    for trace in traces:
        time = trace['time (seconds)'].values
        all_time_data.append(time)
    combined_time = np.concatenate(all_time_data)

    X = all_force_data.reshape(-1, 1)

    n_components = no_states
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=no_iterations, random_state=random_state)
    model.fit(X)
    hidden_states = model.predict(X)

    # Save the model to the specified path
    joblib.dump(model, savepath)
    print(f"Model saved to {savepath}")

    return X, hidden_states, all_time_data, all_force_data, combined_time


def HMM_predict_force(traces, model_filepath):
    # Load the pre-trained model from the specified file path
    model = joblib.load(model_filepath)
    print(f"Model loaded from {model_filepath}")
    
    # Process the traces
    all_force_data = np.concatenate([trace['force_corrected (pN)'].values for trace in traces])

    all_time_data = []
    for trace in traces:
        time = trace['time (seconds)'].values
        all_time_data.append(time)
    combined_time = np.concatenate(all_time_data)

    X = all_force_data.reshape(-1, 1)

    # Predict hidden states using the pre-trained model
    hidden_states = model.predict(X)

    return X, hidden_states, all_time_data, all_force_data, combined_time

def HMM_fit_bp(traces, random_state, no_states, no_iterations):
    np.random.seed(random_state)
    all_force_data = np.concatenate([trace['bp_corrected'].values for trace in traces])

    all_time_data = []
    for trace in traces:
        time = trace['time'].values
        all_time_data.append(time)
    combined_time = np.concatenate(all_time_data)

    X = all_force_data.reshape(-1, 1)

    n_components = no_states
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=no_iterations, random_state=random_state)
    model.fit(X)
    hidden_states = model.predict(X)

    return X, hidden_states, all_time_data, all_force_data, combined_time
def plot_transition_and_HMM_force(hmm_result, traces, n_components):
    X, hidden_states, all_time_data, all_force_data, combined_time = hmm_result[0], hmm_result[1], hmm_result[2], hmm_result[3], hmm_result[4]

    # Calculate mean force for each hidden state
    state_means = np.array([X[hidden_states == i].mean() for i in range(n_components)])

    # Sort states in order of decreasing mean force
    sorted_indices = np.argsort(-state_means)
    sorted_hidden_states = np.zeros_like(hidden_states)
    for new_label, old_label in enumerate(sorted_indices):
        sorted_hidden_states[hidden_states == old_label] = new_label

    state_means_sorted = state_means[sorted_indices]

    # Calculate transition matrix excluding self-transitions and count number of transitions
    transition_counts = np.zeros((n_components, n_components))
    transition_numbers = np.zeros((n_components, n_components))

    for i in range(1, len(sorted_hidden_states)):
        prev_state = sorted_hidden_states[i-1]
        curr_state = sorted_hidden_states[i]
        if prev_state != curr_state:
            transition_counts[prev_state, curr_state] += 1
            transition_numbers[prev_state, curr_state] += 1

    # Normalize transition matrix to get probabilities
    transition_matrix_no_self = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    # Prepare annotations for heatmap
    annotations = np.empty_like(transition_matrix_no_self, dtype=object)
    for i in range(n_components):
        for j in range(n_components):
            annotations[i, j] = f'{transition_matrix_no_self[i, j]:.2f}\n(N={int(transition_numbers[i, j])})'

    # Plot transition matrix
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(transition_matrix_no_self, annot=annotations, cmap='Blues', fmt='', cbar=True,
                xticklabels=[f'State {i+1}' for i in range(n_components)],
                yticklabels=[f'State {i+1}' for i in range(n_components)])
    #plt.title(f'Transition Probability Matrix Excluding Self-Transitions ({experiment})')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.show()

    # Plot original force data and HMM states for each trace separately
    start_idx = 0
    for trace_index, trace in enumerate(traces):
        trace_length = len(trace)
        trace_time = all_time_data[trace_index]
        trace_force = trace['force_corrected (pN)'].values
        
        # Get the portion of hidden states for this trace
        trace_hidden_states = sorted_hidden_states[start_idx:start_idx + trace_length]
        start_idx += trace_length

        step_function = np.zeros_like(trace_force)
        for i in range(n_components):
            step_function[trace_hidden_states == i] = state_means_sorted[i]

        plt.figure(figsize=(7, 3))
        plt.plot(trace_time, trace_force, label=f"Trace {trace_index + 1} Force (pN)", alpha=0.7, color='dimgrey', lw=0.5)
        plt.step(trace_time, step_function, where='mid', label="HMM State Levels", lw=1, color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Force (pN)')
        plt.xlim(25,26)
        plt.legend()
        plt.show()

    # Calculate lifetimes for each state with better handling
    fig, axs = plt.subplots(n_components, 2, figsize=(5, n_components * 2))

    for i in range(n_components):
        # Plot distribution of forces for each state
        axs[i, 0].hist(all_force_data[sorted_hidden_states == i], bins=30, color='blue', alpha=0.7)
        axs[i, 0].set_title(f'Force Distribution for State {i+1}')
        axs[i, 0].set_xlabel('Force (pN)')
        axs[i, 0].set_ylabel('Frequency')
        axs[i, 0].set_xlim(10, 14)

        # Calculate lifetimes for each state
        state_lifetimes = []
        in_state = False
        start_time = None  # Initialize start time as None to track when state starts

        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                # Entering a new state
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                # Leaving the state
                if start_time is not None:  # Ensure start_time was properly set
                    duration = combined_time[j] - start_time
                    if duration >= 0:  # Only add non-negative durations
                        state_lifetimes.append(duration)
                    start_time = None  # Reset start time
                in_state = False

        # Handle any remaining state at the end of the data
        if in_state and start_time is not None:
            duration = combined_time[-1] - start_time
            if duration >= 0:  # Only add non-negative durations
                state_lifetimes.append(duration)

        # Plot distribution of lifetimes for each state
        axs[i, 1].hist(state_lifetimes, bins=50, color='grey', alpha=0.7)
        axs[i, 1].set_title(f'Lifetime Distribution for State {i+1}')
        axs[i, 1].set_xlabel('Lifetime (seconds)')
        axs[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return sorted_hidden_states

def plot_transition_and_HMM_bp(hmm_result, traces, n_components):
    X, hidden_states, all_time_data, all_force_data, combined_time = hmm_result[0], hmm_result[1], hmm_result[2], hmm_result[3], hmm_result[4]

    # Calculate mean force for each hidden state
    state_means = np.array([X[hidden_states == i].mean() for i in range(n_components)])

    # Sort states in order of decreasing mean force
    sorted_indices = np.argsort(-state_means)
    sorted_hidden_states = np.zeros_like(hidden_states)
    for new_label, old_label in enumerate(sorted_indices):
        sorted_hidden_states[hidden_states == old_label] = new_label

    state_means_sorted = state_means[sorted_indices]

    # Calculate transition matrix excluding self-transitions and count number of transitions
    transition_counts = np.zeros((n_components, n_components))
    transition_numbers = np.zeros((n_components, n_components))

    for i in range(1, len(sorted_hidden_states)):
        prev_state = sorted_hidden_states[i-1]
        curr_state = sorted_hidden_states[i]
        if prev_state != curr_state:
            transition_counts[prev_state, curr_state] += 1
            transition_numbers[prev_state, curr_state] += 1

    # Normalize transition matrix to get probabilities
    transition_matrix_no_self = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    # Prepare annotations for heatmap
    annotations = np.empty_like(transition_matrix_no_self, dtype=object)
    for i in range(n_components):
        for j in range(n_components):
            annotations[i, j] = f'{transition_matrix_no_self[i, j]:.2f}\n(N={int(transition_numbers[i, j])})'

    # Plot transition matrix
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(transition_matrix_no_self, annot=annotations, cmap='Blues', fmt='', cbar=True,
                xticklabels=[f'State {i+1}' for i in range(n_components)],
                yticklabels=[f'State {i+1}' for i in range(n_components)])
    #plt.title(f'Transition Probability Matrix Excluding Self-Transitions ({experiment})')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.show()

    # Plot original force data and HMM states for each trace separately
    start_idx = 0
    for trace_index, trace in enumerate(traces):
        trace_length = len(trace)
        trace_time = all_time_data[trace_index]
        trace_force = trace['bp_corrected'].values
        
        # Get the portion of hidden states for this trace
        trace_hidden_states = sorted_hidden_states[start_idx:start_idx + trace_length]
        start_idx += trace_length

        step_function = np.zeros_like(trace_force)
        for i in range(n_components):
            step_function[trace_hidden_states == i] = state_means_sorted[i]

        plt.figure(figsize=(7, 3))
        plt.plot(trace_time, trace_force, label=f"Trace {trace_index + 1}", alpha=0.7, color='dimgrey', lw=1.5)
        plt.step(trace_time, step_function, where='mid', label="HMM State Levels", lw=1.5, color='red')
        plt.xlabel('Time')
        plt.xlim(84.73, 85.15)
        plt.ylim(-45, 75)
        plt.ylabel('bp')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'hmm_trace_{trace_index}.pdf', format='pdf')
        plt.show()

    # Calculate lifetimes for each state with better handling
    fig, axs = plt.subplots(n_components, 2, figsize=(5, n_components * 2))

    for i in range(n_components):
        # Plot distribution of forces for each state
        axs[i, 0].hist(all_force_data[sorted_hidden_states == i], bins=30, color='blue', alpha=0.7, density=True)
        axs[i, 0].set_title(f'Bp Distribution for State {i+1}')
        axs[i, 0].set_xlabel('bp')
        axs[i, 0].set_ylabel('Frequency')
        #axs[i, 0].set_xlim(10, 14)

        # Calculate lifetimes for each state
        state_lifetimes = []
        in_state = False
        start_time = None  # Initialize start time as None to track when state starts

        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                # Entering a new state
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                # Leaving the state
                if start_time is not None:  # Ensure start_time was properly set
                    duration = combined_time[j] - start_time
                    if duration >= 0:  # Only add non-negative durations
                        state_lifetimes.append(duration)
                    start_time = None  # Reset start time
                in_state = False

        # Handle any remaining state at the end of the data
        if in_state and start_time is not None:
            duration = combined_time[-1] - start_time
            if duration >= 0:  # Only add non-negative durations
                state_lifetimes.append(duration)

        # Plot distribution of lifetimes for each state
        axs[i, 1].hist(state_lifetimes, bins=30, color='grey', alpha=0.7, density=True)
        axs[i, 1].set_title(f'Lifetime Distribution for State {i+1}')
        axs[i, 1].set_xlabel('Lifetime (seconds)')
        axs[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return sorted_hidden_states

def exponential_pdf(x, b):
    return b * np.exp(-b * x)

def chi_square_gof(observed, expected):
    observed_sum = np.sum(observed)
    expected_sum = np.sum(expected)
    
    # Normalize expected values to match the sum of observed values
    expected = expected * (observed_sum / expected_sum)
    
    # Calculate chi-square
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat, p_value
def plot_lifetimes_and_force_statistics(n_components, sorted_hidden_states, hmm_result):
    all_force_data, combined_time = hmm_result[3], hmm_result[4]
    # Updated code with additional scatter plot for lifetime at each occurrence
    fig, axs = plt.subplots(n_components, 4, figsize=(12, n_components * 3))  # Added an additional column for cumulative plot

    for i in range(n_components):
        # Get force data for current state
        state_forces = all_force_data[sorted_hidden_states == i]
        
        # Plot histogram of forces for each state
        hist, bin_edges = np.histogram(state_forces, bins=35, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axs[i, 0].hist(state_forces, bins=35, color='dimgrey', edgecolor='black', alpha=0.7, density=True)
        axs[i, 0].set_title(f'Force Distribution for State {i+1}')
        axs[i, 0].set_xlabel('Force (pN)')
        axs[i, 0].set_ylabel('Density')

        # Fit Gaussian to the force distribution
        mu, std = norm.fit(state_forces)
        p_gauss = norm.pdf(bin_centers, mu, std)
        
        # Calculate chi-square for the force distribution fit
        chi2_stat_force, p_value_force = chi_square_gof(hist, p_gauss * np.diff(bin_edges))
        
        # Plot the Gaussian fit with mu and sigma
        axs[i, 0].plot(bin_centers, p_gauss, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.2f}$,\n$\sigma={std:.2f}$\n$\chi^2={chi2_stat_force:.2f}$')
        axs[i, 0].legend(loc='upper left')

        # Calculate lifetimes for each state
        state_lifetimes = []
        occurrence_times = []  # Track time of each occurrence for the scatter plot
        in_state = False
        start_time = None  # Initialize start time as None to track when state starts

        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                # Entering a new state
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                # Leaving the state
                if start_time is not None:  # Ensure start_time was properly set
                    duration = combined_time[j] - start_time
                    if duration >= 0:  # Only add non-negative durations
                        state_lifetimes.append(duration)
                        occurrence_times.append(start_time)  # Track the time of this state occurrence
                    start_time = None  # Reset start time
                in_state = False

        # Handle any remaining state at the end of the data
        if in_state and start_time is not None:
            duration = combined_time[-1] - start_time
            if duration >= 0:  # Only add non-negative durations
                state_lifetimes.append(duration)
                occurrence_times.append(start_time)

        # Convert the list to a NumPy array and filter the lifetimes
        state_lifetimes_array = np.array(state_lifetimes)
        filtered_lifetimes = state_lifetimes_array[state_lifetimes_array < 5]
        filtered_lifetimes = filtered_lifetimes.tolist()

        # Plot histogram of lifetimes for each state
        hist, bin_edges = np.histogram(filtered_lifetimes, bins=500, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axs[i, 1].hist(filtered_lifetimes, bins=120, color='salmon', edgecolor='black', alpha=0.7, density=True)
        axs[i, 1].set_title(f'Lifetime Distribution for State {i+1}')
        axs[i, 1].set_xlabel('Lifetime (seconds)')
        axs[i, 1].set_ylabel('Density')

        # Fit exponential to the lifetime distribution
        if len(filtered_lifetimes) > 0:
            popt, pcov = curve_fit(exponential_pdf, bin_centers, hist, p0=(1,))
            b = popt[0]
            tau = 1 / b *1000  # Lifetime (tau) in milliseconds
            tau_std_error = np.sqrt(np.diag(pcov))[0] / (b**2) * 1000  # Standard error on tau
            
            # Generate expected values for the exponential fit and calculate chi-square
            p_exp = exponential_pdf(bin_centers, b)
            chi2_stat_lifetime, p_value_lifetime = chi_square_gof(hist, p_exp * np.diff(bin_edges))

            # Plot the exponential fit with tau and standard error
            axs[i, 1].plot(bin_centers, p_exp, 'r-', label=f'Exponential Fit\n$\\tau={tau:.2f} \\pm {tau_std_error:.2f}$')
            axs[i, 1].legend()
            axs[i, 1].set_xlim(0, max(filtered_lifetimes) * 0.2)

        # Plot scatter plot for the lifetime of each state at each occurrence
        axs[i, 2].scatter(occurrence_times, state_lifetimes, color='blue', alpha=0.7, s=1)
        axs[i, 2].set_title(f'Lifetime Scatter for State {i+1}')
        axs[i, 2].set_xlabel('Time (seconds)')
        axs[i, 2].set_ylabel('Lifetime (seconds)')

        # Merge timepoints across all traces, removing duplicates
        unique_times, unique_indices = np.unique(combined_time, return_index=True)
        unique_states = sorted_hidden_states[unique_indices]

        # Calculate cumulative counts of the state over unique timepoints
        cumulative_time_in_state = np.cumsum([1 if unique_states[j] == i else 0 for j in range(len(unique_states))])
        normalized_cumulative_time = cumulative_time_in_state / max(cumulative_time_in_state)  # Normalize to 0-1 range

        # Plot cumulative frequency over time
        axs[i, 3].plot(unique_times, normalized_cumulative_time, color='green')
        axs[i, 3].set_title(f'Cumulative Frequency for State {i+1}')
        axs[i, 3].set_xlabel('Time (seconds)')
        axs[i, 3].set_ylabel('Cumulative Frequency')

    plt.savefig('/Users/emily/Desktop/lifetime_and_force_statistics.pdf', format='pdf')
    plt.tight_layout()
    plt.show()

def plot_lifetimes_and_bp_statistics(n_components, sorted_hidden_states, hmm_result):
    all_force_data, combined_time = hmm_result[3], hmm_result[4]
    # Updated code with additional scatter plot for lifetime at each occurrence
    fig, axs = plt.subplots(n_components, 4, figsize=(12, n_components * 3))  # Added an additional column for cumulative plot

    for i in range(n_components):
        # Get force data for current state
        state_forces = all_force_data[sorted_hidden_states == i]
        
        # Plot histogram of forces for each state
        hist, bin_edges = np.histogram(state_forces, bins=35, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axs[i, 0].hist(state_forces, bins=35, color='dimgrey', edgecolor='black', alpha=0.7, density=True)
        axs[i, 0].set_title(f'Bp Distribution for State {i+1}')
        axs[i, 0].set_xlabel('bp')
        axs[i, 0].set_ylabel('Density')

        # Fit Gaussian to the force distribution
        mu, std = norm.fit(state_forces)
        p_gauss = norm.pdf(bin_centers, mu, std)
        
        # Calculate chi-square for the force distribution fit
        chi2_stat_force, p_value_force = chi_square_gof(hist, p_gauss * np.diff(bin_edges))
        
        # Plot the Gaussian fit with mu and sigma
        axs[i, 0].plot(bin_centers, p_gauss, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.2f}$,\n$\sigma={std:.2f}$\n$\chi^2={chi2_stat_force:.2f}$')
        axs[i, 0].legend(loc='upper left')

        # Calculate lifetimes for each state
        state_lifetimes = []
        occurrence_times = []  # Track time of each occurrence for the scatter plot
        in_state = False
        start_time = None  # Initialize start time as None to track when state starts

        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                # Entering a new state
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                # Leaving the state
                if start_time is not None:  # Ensure start_time was properly set
                    duration = combined_time[j] - start_time
                    if duration >= 0:  # Only add non-negative durations
                        state_lifetimes.append(duration)
                        occurrence_times.append(start_time)  # Track the time of this state occurrence
                    start_time = None  # Reset start time
                in_state = False

        # Handle any remaining state at the end of the data
        if in_state and start_time is not None:
            duration = combined_time[-1] - start_time
            if duration >= 0:  # Only add non-negative durations
                state_lifetimes.append(duration)
                occurrence_times.append(start_time)

        # Convert the list to a NumPy array and filter the lifetimes
        state_lifetimes_array = np.array(state_lifetimes)
        filtered_lifetimes = state_lifetimes_array[state_lifetimes_array < 100]
        filtered_lifetimes = filtered_lifetimes.tolist()

        # Plot histogram of lifetimes for each state
        hist, bin_edges = np.histogram(filtered_lifetimes, bins=500, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axs[i, 1].hist(filtered_lifetimes, bins=120, color='salmon', edgecolor='black', alpha=0.7, density=True)
        axs[i, 1].set_title(f'Lifetime Distribution for State {i+1}')
        axs[i, 1].set_xlabel('Lifetime (seconds)')
        axs[i, 1].set_ylabel('Density')

        # Fit exponential to the lifetime distribution
        if len(filtered_lifetimes) > 0:
            popt, pcov = curve_fit(exponential_pdf, bin_centers, hist, p0=(1,))
            b = popt[0]
            tau = 1 / b *1000  # Lifetime (tau) in milliseconds
            tau_std_error = np.sqrt(np.diag(pcov))[0] / (b**2) * 1000  # Standard error on tau
            
            # Generate expected values for the exponential fit and calculate chi-square
            p_exp = exponential_pdf(bin_centers, b)
            chi2_stat_lifetime, p_value_lifetime = chi_square_gof(hist, p_exp * np.diff(bin_edges))

            # Plot the exponential fit with tau and standard error
            axs[i, 1].plot(bin_centers, p_exp, 'r-', label=f'Exponential Fit\n$\\tau={tau:.2f} \\pm {tau_std_error:.2f}$')
            axs[i, 1].legend()
            axs[i, 1].set_xlim(0, max(filtered_lifetimes) * 0.5)

        # Plot scatter plot for the lifetime of each state at each occurrence
        axs[i, 2].scatter(occurrence_times, state_lifetimes, color='blue', alpha=0.7, s=1)
        axs[i, 2].set_title(f'Lifetime Scatter for State {i+1}')
        axs[i, 2].set_xlabel('Time (seconds)')
        axs[i, 2].set_ylabel('Lifetime (seconds)')

        # Merge timepoints across all traces, removing duplicates
        unique_times, unique_indices = np.unique(combined_time, return_index=True)
        unique_states = sorted_hidden_states[unique_indices]

        # Calculate cumulative counts of the state over unique timepoints
        cumulative_time_in_state = np.cumsum([1 if unique_states[j] == i else 0 for j in range(len(unique_states))])
        normalized_cumulative_time = cumulative_time_in_state / max(cumulative_time_in_state)  # Normalize to 0-1 range

        # Plot cumulative frequency over time
        axs[i, 3].plot(unique_times, normalized_cumulative_time, color='green')
        axs[i, 3].set_title(f'Cumulative Frequency for State {i+1}')
        axs[i, 3].set_xlabel('Time (seconds)')
        axs[i, 3].set_ylabel('Cumulative Frequency')

    plt.tight_layout()
    plt.show()

def plot_cdf_lifetime(n_components, sorted_hidden_states, hmm_result, xmin, xmax):
    combined_time = hmm_result[4]

    # Define an exponential cumulative distribution function (CDF) that starts at (0, 0)
    def exponential_cdf(x, b):
        return 1 - np.exp(-b * x)

    # Function to calculate the empirical CDF from binned data
    def calculate_empirical_cdf(hist, bin_edges):
        cdf = np.cumsum(hist) / np.sum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, cdf

    # Prepare to collect lifetimes for each state
    state_lifetimes_lists = []
    results = {"state_lifetimes": [], "bin_centers": [], "empirical_cdfs": [], "fitted_cdfs": [], "taus": [], "tau_errors": []}

    # Iterate over each state to collect lifetimes
    for i in range(n_components):
        state_lifetimes = []
        in_state = False
        start_time = None

        for j in range(len(sorted_hidden_states)):
            if sorted_hidden_states[j] == i and not in_state:
                start_time = combined_time[j]
                in_state = True
            elif sorted_hidden_states[j] != i and in_state:
                if start_time is not None:
                    duration = combined_time[j] - start_time
                    if duration >= 0:
                        state_lifetimes.append(duration)
                    start_time = None
                in_state = False

        # Handle remaining state at the end of the trace
        if in_state and start_time is not None:
            duration = combined_time[-1] - start_time
            if duration >= 0:
                state_lifetimes.append(duration)

        state_lifetimes_lists.append(np.array(state_lifetimes))

    # Initialize the main CDF plot
    fig, ax_cdf = plt.subplots(figsize=(4, 4))

    # Colors and labels for different states
    colors = ['steelblue', 'black', 'crimson', 'dimgrey']
    labels = ['I', 'II', 'III', 'IV']

    taus = []
    taus_errors = []

    # Iterate over each state
    for i, state_lifetimes in enumerate(state_lifetimes_lists):
        # Filter lifetimes to be below 5 seconds for consistency
        filtered_lifetimes = state_lifetimes[state_lifetimes < 5]

        # Ensure there are lifetimes to fit
        if len(filtered_lifetimes) > 0:
            # Create a histogram for the lifetimes to compute a binned empirical CDF
            num_bins = min(50, int(np.sqrt(len(filtered_lifetimes))))
            hist, bin_edges = np.histogram(filtered_lifetimes, bins=num_bins, density=False)
            bin_centers, empirical_cdf = calculate_empirical_cdf(hist, bin_edges)
            
            # Add (0, 0) to the empirical CDF
            bin_centers = np.insert(bin_centers, 0, 0)
            empirical_cdf = np.insert(empirical_cdf, 0, 0)

            # Use histogram values as weights for fitting
            weights = 1 / np.sqrt(hist + 1e-10)
            weights = np.insert(weights, 0, weights[0])  # Extend weights to match CDF length

            # Fit the exponential CDF to the binned empirical CDF using weights
            popt, pcov = curve_fit(exponential_cdf, bin_centers, empirical_cdf, p0=(1,), sigma=weights, absolute_sigma=True)
            b = popt[0]
            tau = 1 / b * 1000  # Calculate tau in milliseconds
            tau_error = np.sqrt(np.diag(pcov))[0] / (b**2) * 1000  # Calculate uncertainty in tau

            taus.append(tau)
            taus_errors.append(tau_error)

            # Generate CDF data from the fit for plotting
            fitted_cdf = exponential_cdf(bin_centers, b)
            
            # Store the results for each state
            results["state_lifetimes"].append(filtered_lifetimes)
            results["bin_centers"].append(bin_centers * 1000)  # Convert to ms
            results["empirical_cdfs"].append(empirical_cdf)
            results["fitted_cdfs"].append(fitted_cdf)
            results["taus"].append(tau)
            results["tau_errors"].append(tau_error)

            # Plot the empirical CDF as a solid line
            ax_cdf.plot(bin_centers * 1000, empirical_cdf, color=colors[i], linewidth=1.5, alpha=0.7)

            # Plot the fitted exponential CDF as a dashed line
            ax_cdf.plot(bin_centers * 1000, fitted_cdf, '--', color=colors[i], linewidth=2)

            # Extend the empirical CDF and fit until 500 ms with constant value of 1
            extended_times = np.arange(bin_centers[-1], 500 / 1000, step=bin_centers[1] - bin_centers[0]) * 1000  # In ms
            extended_cdf = np.ones_like(extended_times)  # Constant value of 1

            ax_cdf.plot(extended_times, extended_cdf, color=colors[i], linewidth=1.5, alpha=0.7)
            ax_cdf.plot(extended_times, extended_cdf, '--', color=colors[i], linewidth=2)

    # Customize the CDF plot
    ax_cdf.set_xlabel('Lifetime (ms)')
    ax_cdf.set_ylabel('Cumulative distribution')
    ax_cdf.set_xlim(xmin, xmax)

    # Create an inset axes for the bar plot
    ax_bar = fig.add_axes([0.44, 0.23, 0.49, 0.49])  # Adjust these coordinates and size as needed
    bars = ax_bar.bar(range(n_components), taus, yerr=taus_errors, capsize=2.5, color=colors[:n_components], alpha=0.7, edgecolor='black')
    ax_bar.set_ylabel('τ (ms)')
    ax_bar.set_xticks(range(n_components))
    ax_bar.set_xticklabels(['I', 'II', 'III', 'IV'][:n_components], rotation=0, fontsize=12)
    ax_bar.set_ylim(0, max(taus) + max(taus_errors) * 2.5)

    # Add tau values on top of the bars
    for bar, tau, error in zip(bars, taus, taus_errors):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, height + max(taus_errors) * 1.2,
                    f'{tau:.2f}', ha='center', va='bottom', fontsize=11)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.savefig('test.pdf', format='pdf')
    plt.show()

    # Return the time, empirical data, fitted CDF, tau, and tau error for all states
    return results
