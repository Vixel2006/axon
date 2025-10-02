import idrak.functions as F
import gc

def run_test():
    # Create a tensor
    t = F.zeros((10, 10))
    print(f"Created tensor with shape: {t.shape}")
    # The tensor 't' will go out of scope here and __del__ should be called

if __name__ == "__main__":
    run_test()
    # Explicitly run garbage collection to help identify reference cycles
    gc.collect()
    print("Test finished. Check Valgrind output for memory leaks.")
