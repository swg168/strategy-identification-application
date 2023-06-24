# strategy-identification-application
## Win 10/11
> **Warning**
> The application has not been tried under MacOS and Linux, and it is not guaranteed to run successfully.
### Download
```sh
git clone https://github.com/swg168/strategy-identification-application.git
cd strategy-identification-application
```
Open the terminal in the repository folder, select the following way to install dependence
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
