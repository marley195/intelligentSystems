# Traffic Flow Prediction

This is a project for Intelligent systems (COS30018) which builds on https://github.com/xiaochus/TrafficFlowPrediction repository of a Traffic flow prediction system

Our repository builds on the Traffic Flow Prediction with Neural Networks to include the follow Models SAEs、LSTM、GRU and RNN.
Using bundooras SCAT data we implement a terminal based system which can calucate the 5 quickest routes between two SCAT locations.

## Requirement

- Python 3.11.10
- Tensorflow 2.13.0
- Keras 2.13.1
- scikit-learn 1.3.1
- MutPy 0.6.1
- numpy 1.24.4
- openyx1 3.15
- xlrd 2.0.1
- folium 0.17.0
- geopy 2.4.1
- graphviz 0.20.3
- matploylib 3.9.2
- networkx 2.7.3
- tkinker 0.1.0

## Setup Virtual Environment

run setup.sh to create and downlaod the required dependencies

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru", "saes" or "simplernn" as arguments. The `.h5` weight file was saved at model folder.

## Experiment

Data are obtained from VicRoads for the city Borondara that containts traffic flow data (the number of cars passing anintersection every 15 minutes).

dataset: PeMS 5min-interval traffic flow data
optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
batch_szie: 256

**Run command below to run the program:**

```
python main.py
Input SCAT Number (beginning position)
Input SCAT Number (Destination position)
```

These are the details for the traffic flow prediction experiment from 200 epoch

| Metrics | MAE  |  MSE   | RMSE |  MAPE  |   R2   | Explained variance score |
| ------- | :--: | :----: | :--: | :----: | :----: | :----------------------: |
| LSTM    | 13.3 | 453.76 | 21.3 | 25.03% | 0.9418 |         0.942165         |
| GRU     | 13.5 | 448.24 | 21.1 | 25.86% | 0.9425 |         0.943976         |
| SAEs    | 13.0 | 393.80 | 19.8 | 23.01% | 0.9495 |         0.949538         |
| RNN     | 13.9 | 425.88 | 20.6 | 26.06% | 0.9454 |         0:948789         |

![evaluate](/images/eva_final.png)

## Reference

    @article{SAEs,
      title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},
      author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
      journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
      year={2015}
    }

    @article{RNN,
      title={Using LSTM and GRU neural network methods for traffic flow prediction},
      author={R Fu, Z Zhang, L Li},
      journal={Chinese Association of Automation, 2017:324-328},
      year={2017}
    }

## Copyright

See [LICENSE](LICENSE) for details.
