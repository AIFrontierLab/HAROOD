from core import train
import sys

# # 1
# filepath = './config/test.yaml'
# results = train(config=filepath)
# print(results)

# # 2
# config_dict = {
#     'algorithm': 'CORAL',
#     'batch_size': 32
# }
# results = train(config=config_dict)
# print(results)

# # 3
results = train(config="./config/test.yaml", lr=1e-3)
print(results)
