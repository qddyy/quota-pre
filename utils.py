import os
from pathlib import Path


def write_env(env_file_path, environ):
    with open(env_file_path, "+w") as file:
        for key, value in environ.items():
            file.write(f"{key}={value}\n")


def read_env(env_file_path):
    environ = {}
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                environ[key] = value
    return environ


if __name__ == "__main__":
    path = Path(__file__).parent
    batch_size = 64
    input_dim = 63
    hidden_dim = 100
    seq_len = 50
    num_layers = 1
    class_num = 5
    environ = {
        "BATCH_SIZE": str(batch_size),
        "INPUT_DIM": str(input_dim),
        "HIDDEN_DIM": str(hidden_dim),
        "SEQ_LEN": str(seq_len),
        "CLASS_NUM": str(class_num),
        "NUM_LAYERS": str(num_layers),
    }
    env_file_path = path / "env_vars.txt"
    write_env(env_file_path, environ)
