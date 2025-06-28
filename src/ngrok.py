# run_with_ngrok.py
import streamlit as st
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError # Thêm import này để xử lý lỗi tốt hơn
import subprocess
import time
import os
import sys

# --- Configuration ---
STREAMLIT_APP_FILE = "src/app.py"  # Your Streamlit app file name
STREAMLIT_PORT = 8501          # Default Streamlit port
NGLO_AUTHTOKEN = "2z5xlYDvqoJe9dmYC01CT7J3jx3_5QjyQVq22MSVpqQoVeV13" # Your ngrok Authtoken

# --- Function to run Streamlit ---
def run_streamlit_app():
    """Run the Streamlit app in a separate process."""
    print(f"[{time.ctime()}] Starting Streamlit app at http://localhost:{STREAMLIT_PORT}")
    
    # Command to run Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP_FILE, "--server.port", str(STREAMLIT_PORT)]
    
    # For Windows, shell=True might be needed to find the "streamlit" command
    if sys.platform == "win32":
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
    return process

# --- Main function ---
def main():
    streamlit_process = None
    ngrok_tunnel = None # Changed variable name for clarity with pyngrok
    
    try:
        # 1. Start Streamlit
        streamlit_process = run_streamlit_app()
        print(f"[{time.ctime()}] Waiting for Streamlit to start...")
        time.sleep(10) # Increased wait time to ensure LLM model has time to load

        # Check if the Streamlit process started successfully
        if streamlit_process.poll() is not None:
            print(f"[{time.ctime()}] Streamlit app might not have started successfully. Checking for errors.")
            stdout, stderr = streamlit_process.communicate()
            print("Streamlit Stdout:\n", stdout.decode())
            print("Streamlit Stderr:\n", stderr.decode())
            return

        # 2. Configure and create ngrok tunnel with pyngrok
        print(f"[{time.ctime()}] Configuring ngrok with Authtoken...")
        try:
            # How to set Authtoken with pyngrok
            ngrok.set_auth_token(NGLO_AUTHTOKEN) 
        except Exception as e:
            print(f"[{time.ctime()}] Error setting ngrok Authtoken: {e}")
            print("Please check your Authtoken.")
            return

        print(f"[{time.ctime()}] Creating ngrok tunnel to localhost:{STREAMLIT_PORT}...")
        try:
            # Use ngrok.connect() for pyngrok
            ngrok_tunnel = ngrok.connect(STREAMLIT_PORT, "http") 
        except PyngrokNgrokError as e:
            print(f"[{time.ctime()}] Ngrok error: {e}")
            print("Ensure the ngrok binary is installed and your Authtoken is correct.")
            print("You might need to run `ngrok authtoken YOUR_NGLO_AUTHTOKEN_HERE` in your terminal.")
            return
        except Exception as e:
            print(f"[{time.ctime()}] Unknown error while creating tunnel: {e}")
            return
        
        # Access public URL via public_url attribute for pyngrok
        public_url = ngrok_tunnel.public_url 
        print(f"[{time.ctime()}] Your Streamlit app is now accessible at: {public_url}")
        print(f"[{time.ctime()}] Press Ctrl+C to stop.")

        # Keep the script running so the tunnel and app don't close
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"[{time.ctime()}] Ctrl+C received. Stopping ngrok tunnel and Streamlit app...")
    except Exception as e:
        print(f"[{time.ctime()}] An error occurred: {e}")
    finally:
        # Ensure ngrok tunnel is closed
        if ngrok_tunnel:
            try:
                ngrok.kill() # For pyngrok, ngrok.kill() closes all tunnels
                print(f"[{time.ctime()}] Ngrok tunnel disconnected.")
            except Exception as e:
                print(f"[{time.ctime()}] Error disconnecting ngrok tunnel: {e}")

        # Ensure Streamlit process is terminated
        if streamlit_process:
            try:
                print(f"[{time.ctime()}] Attempting to terminate Streamlit process...")
                streamlit_process.terminate() # Send terminate signal
                streamlit_process.wait(timeout=5) # Wait a bit for the process to terminate
                if streamlit_process.poll() is None: # If still not terminated
                    streamlit_process.kill() # Force kill
                print(f"[{time.ctime()}] Streamlit process terminated.")
            except Exception as e:
                print(f"[{time.ctime()}] Error terminating Streamlit process: {e}")

if __name__ == "__main__":
    main()