import torch

def check_gpu_availability():
    print(f"Available CUDA: {torch.cuda.is_available()}") 
    if torch.cuda.is_available(): 
        print(f"GPU Name: {torch.cuda.get_device_name(0)}") 
        print(f"Number of GPU(s): {torch.cuda.device_count()}") 
    else: 
        print("CUDA is not available, use CPU instead") 

if __name__ == "__main__":
    check_gpu_availability()