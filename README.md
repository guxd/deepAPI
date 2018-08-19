# Deep API Learning

  Code for the FSE 2016 paper [Deep API Learning](https://guxd.github.io/papers/deepapi.pdf).

## Two Versions
We release both ```Theano``` and ```PyTorch``` code of our approach, in the ```theano``` and ```pytorch``` folders, respectively.

- The ```theano``` folder contains the code to run the experiments presented in the paper. The code is frozen to what it was when we originally wrote the paper. (NOTE: we modified some deprecated API invocations to fit for the latest python and theano).

- The ```PyTorch``` is the bleeding-edge reporitory where we packaged it up, improved the code quality and added some features.

If you are interested in using DeepAPI, check out the PyTorch version and feel free to contribute.

For more information, please refer to the README files under the directory of each component.



## Tool Demo

An online tool demo can be found in http://211.249.63.55/ (Currently shut down due to limited budget)

## Citation
If you find it useful and would like to cite it, the following would be appropriate:
    
    @inproceedings{gu2016deepapi,
        author = {Gu, Xiaodong and Zhang, Hongyu and Zhang, Dongmei and Kim, Sunghun},
        title = {Deep API Learning},
        booktitle = {Proceedings of the 2016 24th ACM SIGSOFT International Symposium on Foundations of Software Engineering},
        series = {FSE 2016},
        year = {2016},
        location = {Seattle, WA, USA},
        pages = {631--642},
        publisher = {ACM},
        address = {New York, NY, USA},
    }