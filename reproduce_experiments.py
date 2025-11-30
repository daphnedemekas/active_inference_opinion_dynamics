import subprocess
import sys
import os

def run_experiment(module_name):
    print(f"Running {module_name}...")
    try:
        subprocess.run([sys.executable, "-m", module_name], check=True)
        print(f"Successfully ran {module_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {module_name}: {e}\n")

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Run demos
    run_experiment("demos.active_inference_demo")
    run_experiment("demos.inference_demo")
    run_experiment("demos.network_demo")

    print("Experiments completed. Check the 'results' directory for plots.")

if __name__ == "__main__":
    main()

