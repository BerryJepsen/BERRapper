# ReadMe



To run the code properly, the following requirements need to be met.

>1.While keeping the directory structure unchanged, place the data file (**final_data.pickle**) to software directory "./Src"



> 2.Code is running with distributed data parallel on 2 GPUs, which requires support `NVIDIA Collective Communication Library (NCCL)`, and in our experiments, `CUDA10.2` is used.



> 3.All the library requirements are listed in file "./requirements.txt", you can use "pip install -r requirements.txt" to install all the packages.



>4.Use instruction "python -m torch.distributed.launch --nproc_per_node=2  MTL_main.py" in shell to run the code.



> 5.Make sure the computer is connected to the internet while running the code.

