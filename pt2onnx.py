import torch

from model import Net


def main():
  pytorch_model = Net()
  pytorch_model.load_state_dict(torch.load('pytorch_model.pt', map_location=torch.device('cpu')))
  pytorch_model.eval()
  dummy_input = torch.zeros(1, 1, 28, 28)
  torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)


if __name__ == '__main__':
  main()