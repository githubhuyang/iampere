class Meta:
    data_dir = "./data" # the folder path for saving data such repair problems, training data, and intermediate pathces.
    csv_dir = "./csv" # the folder path for saving the performance csv file
    n_cpu = 4 # the number of CPUs for parallel processing
    solver_timeout = 600 # time limit in seconds
    lr = 1e-4 # the learning rate for training the GNN model
    epoch_num = 100 # the number epochs in training
    max_mem_cost = 8 # the max memory cost in GB
    train_task_set_name = "maxsat-train-small" # the name of the training problem set
    test_task_set_name = "maxsat-test-small" # the name of the testing problem set
