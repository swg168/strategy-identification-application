# strategy-identification-application
## Win 10/11
> **Warning**
> The application has not been tried under MacOS and Linux, therefore, it is not guaranteed to run normally on these platforms
### 1. Download
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/acd11999-6981-4277-8523-96480a63d6e7)
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/6f92f12d-d607-478e-bfb8-35013da56d4c)

#### Download and unzip, install dependencies.

### 2. Install dependencies

#### Way I: Download and install [anaconda](https://www.anaconda.com/download)/[miniconda](https://docs.conda.io/en/main/miniconda.html)(recommended). 
#### Open the terminal from the unzip folder, and run the following commands in the terminal：
```sh
conda create -n strategy_venv python=3.10
conda activate strategy_venv
python -m pip install -r requirements.txt
```
#### Way II: Download and Install [python](https://www.python.org/downloads/)=3.10.
#### Open the terminal from the unzip folder, and run the following commands in the terminal：
```sh
python -m pip install -r requirements.txt
```
### 3. Run
```sh
python app_strategy.py
```
After running the application, click the "Select" button, select a folder containing the eye movement images, such as the test_image folder in this repository
