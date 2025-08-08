

import torch
import torch.multiprocessing as mp
import time
import threading
import signal
import sys

def worker_gpu(gpu_id, matrix_size=8192, dtype=torch.float16):
    """
    Worker function that runs infinite matrix multiplications on a specific GPU
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    print(f"GPU {gpu_id}: Starting worker with matrix size {matrix_size}x{matrix_size}")
    
    # Allocate large matrices to fill GPU memory
    try:
        # Create multiple large matrices to consume memory
        matrices = []
        for i in range(4):  # Create 4 large matrices per GPU
            A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            B = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            matrices.append((A, B))
        
        # Allocate additional result matrices
        results = []
        for i in range(2):
            result = torch.zeros(matrix_size, matrix_size, dtype=dtype, device=device)
            results.append(result)
        
        print(f"GPU {gpu_id}: Allocated {len(matrices)} matrix pairs")
        
        iteration = 0
        start_time = time.time()
        
        while True:
            # Perform multiple matrix multiplications
            for i, (A, B) in enumerate(matrices):
                # Matrix multiplication
                C = torch.matmul(A, B)
                
                # Add some additional operations to increase utilization
                C = torch.relu(C)
                C = C + 0.01 * torch.randn_like(C)
                
                # Store result (rotating through result buffers)
                results[i % len(results)] = C
                
                # Occasionally update one of the input matrices
                if iteration % 100 == 0:
                    A.copy_(torch.randn_like(A))
            
            iteration += 1
            
            # Print progress every 1000 iterations
            if iteration % 1000 == 0:
                elapsed = time.time() - start_time
                ops_per_sec = iteration / elapsed
                print(f"GPU {gpu_id}: {iteration:6d} iterations, {ops_per_sec:.1f} ops/sec")
                
    except torch.cuda.OutOfMemoryError:
        print(f"GPU {gpu_id}: Out of memory with matrix size {matrix_size}, trying smaller size")
        # Try with smaller matrices
        return worker_gpu(gpu_id, matrix_size // 2, dtype)
    except Exception as e:
        print(f"GPU {gpu_id}: Error - {e}")

def monitor_gpu_usage():
    """Monitor and print GPU usage statistics"""
    while True:
        try:
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                print(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
            print("-" * 50)
            time.sleep(10)  # Update every 10 seconds
        except Exception as e:
            print(f"Monitor error: {e}")
            break

def signal_handler(sig, frame):
    print("\nShutting down...")
    sys.exit(0)

def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    # Print GPU information
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}, {memory_gb:.1f}GB memory")
    
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    # Determine matrix size based on GPU memory
    # Start with a large size and let each worker adjust if needed
    base_matrix_size = 32768  # Adjust this based on your GPU memory
    
    # Use float16 to fit larger matrices in memory while maintaining high utilization
    dtype = torch.float16
    
    print(f"Starting workers with base matrix size: {base_matrix_size}")
    print("Press Ctrl+C to stop")
    
    # Start worker processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_gpu, args=(gpu_id, base_matrix_size, dtype))
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger startup
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_gpu_usage, daemon=True)
    monitor_thread.start()
    
    try:
        # Wait for all processes
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nTerminating workers...")
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    main()
