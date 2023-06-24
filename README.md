# strategy-identification-application
## Win 10/11
> **Warning**
> The application has not been tried under MacOS and Linux, and it is not guaranteed to run successfully.
### Download
```sh
if you had git bash:
git clone https://github.com/swg168/strategy-identification-application.git
cd strategy-identification-application
else you can
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/72f018ba-7edc-4510-bb0c-34a8454b7b95)
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/611b8c14-f690-40b4-b61e-cd41efdfbec3)
Unzip after downloading，
Open the terminal in the repository folder, select the following way to install dependence
```

### Installation dependence
```sh
# Way I: Install python=3.10
python -m pip install -r requirements.txt   

# Way II: anaconda/miniconda(recommended)：
conda create -n strategy_venv python=3.10
conda activate strategy_venv
python -m pip install -r requirements.txt
```

### Run
```sh
python app_strategy.py
```
After running the application, click the "Select" button, select a folder containing the eye movement images, such as the test_image folder in this repository
