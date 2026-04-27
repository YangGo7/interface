Release package usage

Version: 0.0.0

For non-technical users

1. Double-click 0-install-and-run.bat
2. Wait until the install finishes and the server starts
3. Open http://localhost:5000

Manual mode

1. Run .\setup_backend.ps1
2. Edit gpts\.env if you need GEMINI_API_KEY or GPU mode
3. Run .\start_server.ps1
4. Open http://localhost:5000

GPU mode

- Run .\start_server_gpu.ps1 or double-click 3-run-gpu.bat
- If CUDA is not available, the server falls back to CPU
