"""
Sample Code for Timer Usage
"""

from timer import Timer

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

def main(args):
    
    # ...

    # Timer
    timer = Timer(verbosity_level=configs["log_verbosity"], log_fn=metric)

    for epoch in range(configs["num_epochs"]):
        
        # ...
        
        with timer("train_loading"):
            train_loader = task.get_train_iterator()
        
        for batch_idx, batch in enumerate(train_loader):
            # ...
            
            # (BASIC) Compute a SG g
            with timer("forward.backward"):
                loss, grads = task.batch_loss_grad(batch)
                train_loss += loss.item()

            with timer("grad.exchange"):
                # ...

            """Below Code is common in all distributed learning"""

            with timer("grad.step"):
                # ...

    print(timer.summary())

