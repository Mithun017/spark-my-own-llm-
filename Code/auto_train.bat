@echo off
echo ========================================================
echo             SPARK SLM AUTO-TRAINING PIPELINE
echo ========================================================
echo.

:: Ensure we are in the correct directory (where the .bat file is located)
cd /d "%~dp0"

echo [1/6] Running Stage 1: Data Collection...
python src\data_pipeline\collector.py
if %errorlevel% neq 0 (
    echo Error during Data Collection.
    pause
    exit /b %errorlevel%
)
echo.

echo [2/6] Running Stage 1: Data Cleaning...
python src\data_pipeline\cleaner.py
if %errorlevel% neq 0 (
    echo Error during Data Cleaning.
    pause
    exit /b %errorlevel%
)
echo.

echo [3/6] Running Stage 2: Tokenizer Training...
python src\tokenizer\bpe_tokenizer.py
if %errorlevel% neq 0 (
    echo Error during Tokenizer Training.
    pause
    exit /b %errorlevel%
)
echo.

echo [4/6] Running Stage 3: Neural Network Pretraining...
python src\training\pretrain.py
if %errorlevel% neq 0 (
    echo Error during Base Pretraining.
    pause
    exit /b %errorlevel%
)
echo.

echo [5/6] Running Stage 4: Instruction Fine-Tuning...
python src\training\finetune.py
if %errorlevel% neq 0 (
    echo Error during Instruction Fine-Tuning.
    pause
    exit /b %errorlevel%
)
echo.

echo [6/6] Running Stage 5: Evaluation ^& Benchmarking...
python src\evaluation\evaluate.py
if %errorlevel% neq 0 (
    echo Error during Evaluation.
    pause
    exit /b %errorlevel%
)
echo.

echo ========================================================
echo        SPARK SLM AUTO-TRAINING COMPLETELY FINISHED
echo ========================================================
echo.
echo The model is now fully trained and ready to talk!
echo To chat with your custom model, navigate to the Final_output
echo folder and start the Streamlit app:
echo.
echo cd Final_output
echo streamlit run app.py
echo.
pause
