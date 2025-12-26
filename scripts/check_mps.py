import torch

def main():
    print("torch:", torch.__version__)
    print("mps available:", torch.backends.mps.is_available())
    print("mps built:", torch.backends.mps.is_built())
    if torch.backends.mps.is_available():
        x = torch.randn(2, 3, 64, 64, device="mps")
        y = x * 2
        print("mps tensor ok:", y.mean().item())

if __name__ == "__main__":
    main()
