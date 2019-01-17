# Deep API Learning

Theano implementation of DeepAPI ([Deep API Learning](https://arxiv.org/abs/1605.08535))

## Dependency

* Python 3.6
* Theano 1.0
 ```bash
   pip install -r requirements.txt
   cd 
   vim ~/.theanorc
```
```
[global]
floatX=float32
device=cuda0
[gpuarray]
preallocate = 1
```


## Runing
### Download Dataset:
download data from [Google Driver](https://drive.google.com/drive/folders/1jBKMWZr5ZEyLaLgH34M7AjJ2v52Cq5vv?usp=sharing) and save them to the `./data` folder

### Configuration:
Edit hyperparameters in the bottom of `state.py` file.

### Train
```bash
    python train.py --state proto_search_state
```
    
### Test
 ```bash
    python sample.py
 ```
