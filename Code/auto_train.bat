@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "LOGFILE=..\Log.txt"

call :log "========================================================"
call :log "           SPARK LLM MASSIVE-SCALE PIPELINE"
call :log "========================================================"
call :log "_BLANK_"

call :log "[1/8] Streaming Extrapolated Dataset Sample for Vocabulary [Stage 12]..."
python src\data_pipeline\hf_collector.py
if %errorlevel% neq 0 (
    call :log "Error during Data Collection."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[2/8] Training BPE Tokenizer..."
python src\tokenizer\bpe_tokenizer.py
if %errorlevel% neq 0 (
    call :log "Error during Tokenizer Training."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[3/8] Compiling High-Velocity Binary Dataset Memmap [Stage 12, Part 2]..."
python src\data_pipeline\hf_build_bin.py
if %errorlevel% neq 0 (
    call :log "Error during Binary Data Compilation."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[4/8] Running Deep Neural Pretraining [with Checkpointing]..."
python src\training\pretrain.py
if %errorlevel% neq 0 (
    call :log "Error during Base Pretraining."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[5/8] Running Instruction Fine-Tuning..."
python src\training\finetune.py
if %errorlevel% neq 0 (
    call :log "Error during Instruction Fine-Tuning."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[6/8] Running Evaluation and Benchmarking..."
python src\evaluation\evaluate.py
if %errorlevel% neq 0 (
    call :log "Error during Evaluation."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[7/8] Running Reinforcement Learning [RLHF]..."
python src\training\rlhf_tune.py
if %errorlevel% neq 0 (
    call :log "Error during Reinforcement Tuning."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "[8/8] Securing Output to Final_output\model..."
xcopy /Y "c:\Users\MITHUN\Desktop\STUDIES\PROJECT\42.SPARK - My own slm\Code\src\model\*.pt" "c:\Users\MITHUN\Desktop\STUDIES\PROJECT\42.SPARK - My own slm\Code\Final_output\model\"
if %errorlevel% neq 0 (
    call :log "Error saving the model."
    pause
    exit /b %errorlevel%
)
call :log "_BLANK_"

call :log "========================================================"
call :log "      SPARK LLM AUTO-TRAINING COMPLETELY FINISHED"
call :log "========================================================"
call :log "_BLANK_"
call :log "The massive-scale model is now fully trained and ready to talk!"
echo To chat with your custom model, navigate to the Final_output
echo folder and start the Streamlit app:
echo.
echo cd Final_output
echo streamlit run app.py
echo.
pause
exit /b

:log
set "msg=%~1"
if "!msg!"=="_BLANK_" (
    echo.
    echo. >> "%LOGFILE%"
) else (
    echo %~1
    echo %~1 >> "%LOGFILE%"
)
exit /b
