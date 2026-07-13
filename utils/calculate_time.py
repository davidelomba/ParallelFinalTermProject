import pandas as pd

def calculate_average_execution_times(file_path):
    """
    Loads benchmark data from a CSV file, calculates the average execution time
    per phase for each strategy across all windows, and returns a formatted DataFrame.
    """
    # Load the benchmark dataset from the CSV file
    df = pd.read_csv(file_path)

    # Convert execution time columns to numeric values.
    # The errors='coerce' argument automatically converts any "n/a" strings into NaN.
    df['candidate_mean_s'] = pd.to_numeric(df['candidate_mean_s'], errors='coerce')
    df['baseline_mean_s'] = pd.to_numeric(df['baseline_mean_s'], errors='coerce')

    # Calculate the mean execution time across all windows for each parallel strategy (candidate)
    candidates_avg = df.groupby(['candidate', 'phase'])['candidate_mean_s'].mean().reset_index()
    candidates_avg.rename(columns={'candidate': 'strategy', 'candidate_mean_s': 'avg_time_s'}, inplace=True)

    # Calculate the mean execution time across all windows for the sequential baseline.
    # Since the baseline values are duplicated across candidates, grouping by 'phase' is sufficient.
    baseline_avg = df.groupby('phase')['baseline_mean_s'].mean().reset_index()
    baseline_avg['strategy'] = 'sequential'
    baseline_avg.rename(columns={'baseline_mean_s': 'avg_time_s'}, inplace=True)

    # Combine both sequential and parallel strategy metrics into a single DataFrame
    combined_df = pd.concat([baseline_avg, candidates_avg], ignore_index=True)

    # Pivot the table to structure it cleanly: strategies on rows, pipeline phases on columns
    pivot_table = combined_df.pivot(index='strategy', columns='phase', values='avg_time_s')

    # Reorder the columns to match the chronological execution order of the stitching pipeline
    pipeline_order = ['extract', 'match', 'homo', 'warp', 'reext', 'total']
    pivot_table = pivot_table.reindex(columns=pipeline_order)

    return pivot_table


if __name__ == "__main__":
    # Define the path to your CSV data file
    csv_file_path = "results/benchmark_results_4c_4s.csv"
    
    # Optional: you can wrap this in a try-except block to handle missing files gracefully
    try:
        # Call the function and get the processed table
        results_table = calculate_average_execution_times(csv_file_path)
        
        # Display the aggregated benchmark results rounded to 4 decimal places
        print("Average Execution Time (seconds) per Phase across all Windows:")
        print(results_table.round(4))

        output_file = "results/average_results_4c_4s.csv"
        results_table.to_csv(output_file)
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")