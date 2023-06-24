# strategy-identification-application
## Win 10/11
> **Warning**
> The application has not been tried under MacOS and Linux, and it is not guaranteed to run successfully.
### Download
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/acd11999-6981-4277-8523-96480a63d6e7)
![image](https://github.com/swg168/strategy-identification-application/assets/109449633/6f92f12d-d607-478e-bfb8-35013da56d4c)

#### Unzip after downloading，open the terminal in the repository folder, select the following way to install dependence
### Installation dependence
```sh
# Way I: anaconda/miniconda(recommended)：
conda create -n strategy_venv python=3.10
conda activate strategy_venv
python -m pip install -r requirements.txt

# Way II: Install python=3.10
python -m pip install -r requirements.txt
```

### Run
```sh
python app_strategy.py
```
After running the application, click the "Select" button, select a folder containing the eye movement images, such as the test_image folder in this repository
