# strategy-identification-application
### 1. Download
```sh
git clone https://github.com/swg168/strategy-identification-application.git
cd strategy-identification-application
```
Open the terminal in the repository folder, select the following way to install dependence
### 2. Installation dependence
```sh
# Way I
python -m pip install -r requirements.txt   

# Way II (anaconda/miniconda)：
conda create -n strategy_venv python=3.10
conda activate gptac_venv
python -m pip install -r requirements.txt
```

### 3. Run
```sh
python app_strategy.py
```
After running the application, click the "Select" button, select a folder containing the eye movement images, such as the test_image folder in this repository
