import torch

def resolve_device(device_arg: str) -> torch.device:
    """
    --device の指定を柔軟に解釈:
      - "cpu" → cpu
      - "cuda" → cuda (現在のデフォGPU)
      - "0" / "1" / "2" ... → cuda:{index}
      - "cuda:0" / "cuda:1" ... → そのまま
    使えるGPUが無い場合は自動で cpu にフォールバック。
    """
    s = str(device_arg).strip().lower()
    if s == "cuda":
        return torch.device("cpu")
    if s == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if s.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(s)}")
        return torch.device("cpu")
    if s.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(s)
        return torch.device("cpu")
    return torch.device("cpu")

if __name__ == "__main__":
    device = resolve_device(args.device)